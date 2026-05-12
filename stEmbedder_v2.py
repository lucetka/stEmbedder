# fixed version with spinner added

# Lucie's Streamlit embedder
# Original: 16-Aug-2022
# Updated: 24-Mar-2026 (cloud-safe downloads, SPECTER2 adapters, Windows symlink workaround)
#
# Key changes:
# - No server-side file writes required: uses st.download_button
# - Two-step workflow: PREPARE FILES -> Generate embeddings
# - Option A: API-first on Cloud; local models only if dependencies are available
# - Allows "no abstract" via <none> option
# - Windows workaround for HF symlink privilege errors (no admin rights needed)

import io
import os
import platform
import zipfile
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
import streamlit as st

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder

# -----------------------------------------------------------------------------
# Windows symlink workaround (Hugging Face cache)
# -----------------------------------------------------------------------------
# On Windows, Hugging Face cache may try to create symlinks. If Developer Mode/admin
# is not available, this can fail with: "A required privilege is not held by the client".
# We provide an opt-in workaround that downloads to a local directory without symlinks.
IS_WINDOWS = platform.system().lower().startswith("win")

# -----------------------------------------------------------------------------
# Optional local embedding dependencies (Option A)
# -----------------------------------------------------------------------------
HAVE_TORCH_TRANSFORMERS = False
HAVE_ADAPTERS = False

try:
    import torch  # noqa: F401
    from transformers import AutoTokenizer, AutoModel  # noqa: F401
    HAVE_TORCH_TRANSFORMERS = True
except Exception:
    HAVE_TORCH_TRANSFORMERS = False

try:
    from adapters import AutoAdapterModel  # noqa: F401
    HAVE_ADAPTERS = True
except Exception:
    HAVE_ADAPTERS = False

# -----------------------------------------------------------------------------
# Remote SPECTER API config (hosted backend)
# -----------------------------------------------------------------------------
URL_SPECTER_V1 = "https://model-apis.semanticscholar.org/specter/v1/invoke"
MAX_BATCH_SIZE = 16


def chunks(lst, chunk_size=MAX_BATCH_SIZE):
    """Splits a longer list to respect batch size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def embed_api_specter_v1(papers: List[dict]) -> Dict[str, List[float]]:
    """Embeds using Semantic Scholar hosted SPECTER v1 API."""
    embeddings_by_paper_id: Dict[str, List[float]] = {}

    for chunk in chunks(papers):
        #response = requests.post(URL_SPECTER_V1, json=chunk, timeout=180)
        try:
            response = requests.post(URL_SPECTER_V1, json=chunk, timeout=(10, 60))
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"SPECTER v1 API request failed: {e}")
        if response.status_code != 200:
            raise RuntimeError(
                f"SPECTER API error {response.status_code}: {response.text[:300]}"
            )

        for paper in response.json()["preds"]:
            embeddings_by_paper_id[paper["paper_id"]] = paper["embedding"]

    return embeddings_by_paper_id


# -----------------------------------------------------------------------------
# Local Hugging Face backends (enabled only if deps present)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_hf_specter(model_name: str = "allenai/specter", use_no_symlinks: bool = False):
    # Optional Windows workaround: download to local dir without symlinks
    if use_no_symlinks and IS_WINDOWS:
        try:
            from huggingface_hub import snapshot_download
            local_dir = os.path.join(os.getcwd(), "hf_models", model_name.replace("/", "__"))
            snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)
            model_name = local_dir
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device


@st.cache_resource(show_spinner=False)
def _load_hf_specter2(adapter_kind: str = "proximity", use_no_symlinks: bool = False):
    """Load SPECTER2 base + selected adapter.

    adapter_kind:
      - proximity (retrieval / similarity)
      - classification
      - regression
      - adhoc_query (encode short queries)
    """

    base_id = "allenai/specter2_base"
    if use_no_symlinks and IS_WINDOWS:
        try:
            from huggingface_hub import snapshot_download
            base_dir = os.path.join(os.getcwd(), "hf_models", base_id.replace("/", "__"))
            snapshot_download(repo_id=base_id, local_dir=base_dir, local_dir_use_symlinks=False)
            base_id = base_dir
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = AutoAdapterModel.from_pretrained(base_id)

    adapter_map = {
        "proximity": "allenai/specter2",
        "classification": "allenai/specter2_classification",
        "regression": "allenai/specter2_regression",
        "adhoc_query": "allenai/specter2_adhoc_query",
    }
    adapter_id = adapter_map.get(adapter_kind, "allenai/specter2")

    if use_no_symlinks and IS_WINDOWS:
        try:
            from huggingface_hub import snapshot_download
            adapter_dir = os.path.join(os.getcwd(), "hf_models", adapter_id.replace("/", "__"))
            snapshot_download(repo_id=adapter_id, local_dir=adapter_dir, local_dir_use_symlinks=False)
            adapter_id = adapter_dir
        except Exception:
            pass

    model.load_adapter(adapter_id, source="hf", load_as=adapter_kind, set_active=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device


def _hf_embed_dataframe(
    df: pd.DataFrame,
    tokenizer,
    model,
    device,
    batch_size: int = 32,
    max_length: int = 512,
) -> Dict[str, List[float]]:
    """Return embeddings as dict[paper_id] -> vector."""  

    if "paper_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "paper_id"})  

    titles = df.get("title", pd.Series([""] * len(df))).fillna("").astype(str).tolist()  
    abstracts = df.get("abstract", pd.Series([""] * len(df))).fillna("").astype(str).tolist()  

    texts = [
        t + tokenizer.sep_token + a if a.strip() else t
        for t, a in zip(titles, abstracts)
    ]  

    embeddings_by_paper_id: Dict[str, List[float]] = {}  

    # --- PATCH E starts here: progress reporting ---
    total = len(texts)
    progress = st.progress(0)
    status = st.empty()
    # --- PATCH E ends here ---

    for i in range(0, len(texts), batch_size):  
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(device) 

        with torch.no_grad():
            out = model(**inputs)  

        vecs = out.last_hidden_state[:, 0, :].detach().cpu().numpy()  
        batch_ids = df["paper_id"].iloc[i : i + len(batch_texts)].astype(str).tolist()  

        for pid, v in zip(batch_ids, vecs):  
            embeddings_by_paper_id[pid] = v.astype(float).tolist()  # 

        # --- PATCH E starts here: update progress each batch ---
        done = min(total, i + len(batch_texts))
        progress.progress(done / total if total else 1.0)
        status.write(f"Embedded {done:,} / {total:,} documents…")
        # --- PATCH E ends here ---

    status.write("Embedding batches finished.")
    return embeddings_by_paper_id  

def embed_local_hf_specter(df: pd.DataFrame) -> Dict[str, List[float]]:
    use_no_symlinks = bool(st.session_state.get("use_windows_no_symlinks", False))
    tokenizer, model, device = _load_hf_specter("allenai/specter", use_no_symlinks=use_no_symlinks)
    return _hf_embed_dataframe(df, tokenizer, model, device)


def embed_local_hf_specter2(df: pd.DataFrame, adapter_kind: str) -> Dict[str, List[float]]:
    use_no_symlinks = bool(st.session_state.get("use_windows_no_symlinks", False))
    tokenizer, model, device = _load_hf_specter2(adapter_kind, use_no_symlinks=use_no_symlinks)
    return _hf_embed_dataframe(df, tokenizer, model, device)


# -----------------------------------------------------------------------------
# Preparation step (no disk I/O)
# -----------------------------------------------------------------------------

def prepare_input(
    df: pd.DataFrame,
    unique_ID_col: str,
    title_col: str,
    abstract_col: str,
    additional_cols_list: List[str],
):
    """Prepare two aligned dataframes:

    - input_df: paper_id, title, abstract
    - ref_df:  unique ID + additional metadata columns

    Returns: (input_df, ref_df)
    """

    cols = [unique_ID_col, title_col] + additional_cols_list
    if abstract_col != "<none>":
        cols.insert(2, abstract_col)

    work = df[cols].copy()
    work = work.dropna(subset=[unique_ID_col])
    work = work.drop_duplicates(subset=[unique_ID_col], keep="last")
    work = work.reset_index(drop=True)

    ref_cols = [unique_ID_col] + [c for c in additional_cols_list if c != unique_ID_col]
    ref_df = work[ref_cols].copy()

    input_df = work[[unique_ID_col, title_col]].copy()
    input_df = input_df.rename(columns={unique_ID_col: "paper_id", title_col: "title"})

    if abstract_col == "<none>":
        input_df["abstract"] = ""
    else:
        input_df["abstract"] = work[abstract_col].fillna("").astype(str)

    input_df["paper_id"] = input_df["paper_id"].astype(str)
    input_df["title"] = input_df["title"].fillna("").astype(str)

    return input_df, ref_df


def df_to_csv_bytes(df: pd.DataFrame, index: bool = False) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=index)
    return buf.getvalue().encode("utf-8")


def build_zip(files: Dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return out.getvalue()


# -----------------------------------------------------------------------------
# Embedding step (no disk I/O)
# -----------------------------------------------------------------------------

def create_embeddings(
    input_df: pd.DataFrame,
    backend: str,
    specter2_adapter: str = "proximity",
):
    """Return embeddings_df (rows aligned to input_df) and title_index_df."""

    if backend == "SPECTER v1 (API) — hosted (Semantic Scholar)":
        papers = input_df[["paper_id", "title", "abstract"]].to_dict(orient="records")
        all_embeddings = embed_api_specter_v1(papers)

    elif backend == "SPECTER (local HF) — allenai/specter":
        if not HAVE_TORCH_TRANSFORMERS:
            raise RuntimeError("Local HF backend not available (torch/transformers not installed).")
        all_embeddings = embed_local_hf_specter(input_df)

    elif backend == "SPECTER2 (local HF + adapter)":
        if not (HAVE_TORCH_TRANSFORMERS and HAVE_ADAPTERS):
            raise RuntimeError("SPECTER2 backend not available (torch/transformers/adapters not installed).")
        all_embeddings = embed_local_hf_specter2(input_df, specter2_adapter)

    else:
        raise RuntimeError(f"Unknown backend: {backend}")

    emb_df = pd.DataFrame(all_embeddings).T
    emb_df.index.name = "paper_id"
    emb_df = emb_df.reindex(input_df["paper_id"].astype(str).values)

    title_df = emb_df.copy()
    title_df["title"] = input_df["title"].values
    title_df = title_df.set_index("title")

    return emb_df, title_df


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

st.title("Create embeddings for scientific papers (cloud-safe)")

st.write(
    "This app prepares an embedding input file and a matching reference/metadata file, "
    "then generates embeddings using SPECTER (API) and optionally local models if installed."
)

uploaded_file = st.file_uploader("Select your CSV file:", type=["csv"])
if uploaded_file is None:
    st.stop()

source_df = pd.read_csv(uploaded_file)

st.write("Preview:")
gb = GridOptionsBuilder.from_dataframe(source_df, min_column_width=120)
AgGrid(source_df.head(5), gridOptions=gb.build(), fit_columns_on_grid_load=True)

all_columns = list(source_df.columns)

unique_ID_column = st.selectbox("Select unique ID column", all_columns)
title_column = st.selectbox("Select title column", all_columns)
abstract_column = st.selectbox("Select abstract column (or <none>)", ["<none>"] + all_columns)

st.write("Select additional columns for the reference file (year, journal, citations, etc.).")
additional_columns = st.multiselect("Additional columns for reference file:", all_columns)

st.markdown("### Embedding model")

backend_options = ["SPECTER v1 (API) — hosted (Semantic Scholar)"]
if HAVE_TORCH_TRANSFORMERS:
    backend_options.append("SPECTER (local HF) — allenai/specter")
if HAVE_TORCH_TRANSFORMERS and HAVE_ADAPTERS:
    backend_options.append("SPECTER2 (local HF + adapter)")

# Defaults must be set BEFORE widgets that use these keys
if "embedding_backend" not in st.session_state:
    st.session_state["embedding_backend"] = backend_options[0]
if "specter2_adapter" not in st.session_state:
    st.session_state["specter2_adapter"] = "proximity"

embedding_backend = st.selectbox(
    "Choose embedding backend",
    backend_options,
    key="embedding_backend",
)

st.caption(
    "API-first: the hosted SPECTER v1 option always works on Streamlit Cloud. "
    "Local options appear only if the required packages are installed."
)

# Windows workaround toggle (no admin rights)
if IS_WINDOWS:
    use_windows_no_symlinks = st.checkbox(
        "Windows fix: avoid Hugging Face symlinks (recommended if you see 'required privilege' error)",
        value=True,
        help=(
            "If enabled, the app will download HF models into ./hf_models/ without using symlinks. "
            "This helps on Windows machines where symlinks are blocked by policy."
        ),
    )
    st.session_state["use_windows_no_symlinks"] = bool(use_windows_no_symlinks)
    if use_windows_no_symlinks:
        # Also suppress the symlink warning message (not the core fix)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# SPECTER2 adapter selector (only shown if relevant)
specter2_adapter = st.session_state.get("specter2_adapter", "proximity")
if embedding_backend == "SPECTER2 (local HF + adapter)":
    st.caption(
        "SPECTER2 uses task-specific adapters: proximity (retrieval/similarity), classification, regression, "
        "or adhoc_query (encode short queries)."
    )
    specter2_adapter = st.selectbox(
        "SPECTER2 adapter (task format)",
        ["proximity", "classification", "regression", "adhoc_query"],
        index=0,
        key="specter2_adapter",
    )

st.write("## Step 1: Prepare aligned files")
if st.button("PREPARE FILES"):
    try:
        input_df, ref_df = prepare_input(
            source_df,
            unique_ID_col=unique_ID_column,
            title_col=title_column,
            abstract_col=abstract_column,
            additional_cols_list=additional_columns,
        )

        st.session_state["prepared_input_df"] = input_df
        st.session_state["prepared_ref_df"] = ref_df
        st.success(f"Prepared {len(input_df):,} rows (after dropping missing IDs and duplicates).")

        input_bytes = df_to_csv_bytes(input_df, index=False)
        ref_bytes = df_to_csv_bytes(ref_df, index=False)

        base = uploaded_file.name[:-4]

        st.download_button(
            "Download input4specter.csv",
            data=input_bytes,
            file_name=f"{base}_input4specter.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download index_reference.csv",
            data=ref_bytes,
            file_name=f"{base}_index_reference.csv",
            mime="text/csv",
        )

        zip_bytes = build_zip(
            {
                f"{base}_input4specter.csv": input_bytes,
                f"{base}_index_reference.csv": ref_bytes,
            }
        )
        st.download_button(
            "Download both prepared files (ZIP)",
            data=zip_bytes,
            file_name=f"{base}_prepared_files.zip",
            mime="application/zip",
        )

    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        st.error(message, icon="🚨")


st.write("## Step 2: Generate embeddings")
if "prepared_input_df" in st.session_state:
    if st.button("Generate embeddings!"):
        try:
            input_df = st.session_state["prepared_input_df"]
            ref_df = st.session_state["prepared_ref_df"]
            backend = st.session_state.get(
                "embedding_backend", "SPECTER v1 (API) — hosted (Semantic Scholar)"
            )
            adapter = st.session_state.get("specter2_adapter", "proximity")

            ###DEBUG
            
            # 🔴 
            st.session_state["embed_runs"] = st.session_state.get("embed_runs", 0) + 1
            st.write("Embedding run count this session:", st.session_state["embed_runs"])
            
            # --- PATCH D starts here: debug + spinner ---
            st.write("Backend actually used:", backend)
            if backend == "SPECTER2 (local HF + adapter)":
                st.write("SPECTER2 adapter:", adapter)

            import time
            t0 = time.time()

            with st.spinner(f"Generating embeddings using: {backend}"):
                emb_df, title_df = create_embeddings(
                    input_df=input_df,
                    backend=backend,
                    specter2_adapter=adapter,
                )

            st.success(f"Done in {time.time() - t0:.1f} seconds.")
            # --- PATCH D ends here ---


            st.write(f"Embedding {len(input_df):,} rows using: **{backend}**")

            emb_df, title_df = create_embeddings(
                input_df, backend=backend, specter2_adapter=adapter
            )

            base = uploaded_file.name[:-4]

            emb_bytes = df_to_csv_bytes(emb_df.reset_index(), index=False)
            title_bytes = df_to_csv_bytes(title_df.reset_index(), index=False)
            ref_bytes = df_to_csv_bytes(ref_df, index=False)

            st.success("Embeddings created.")

            st.download_button(
                "Download embeddings.csv",
                data=emb_bytes,
                file_name=f"{base}_embeddings.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download embeddings_with_title.csv",
                data=title_bytes,
                file_name=f"{base}_embeddings_with_title.csv",
                mime="text/csv",
            )

            zip_all = build_zip(
                {
                    f"{base}_input4specter.csv": df_to_csv_bytes(input_df, index=False),
                    f"{base}_index_reference.csv": ref_bytes,
                    f"{base}_embeddings.csv": emb_bytes,
                    f"{base}_embeddings_with_title.csv": title_bytes,
                }
            )

            st.download_button(
                "Download ALL outputs (ZIP)",
                data=zip_all,
                file_name=f"{base}_all_outputs.zip",
                mime="application/zip",
            )

        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            st.error(message, icon="🚨")
else:
    st.info("Run PREPARE FILES first.")
