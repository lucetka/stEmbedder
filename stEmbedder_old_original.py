#Lucie's first attempt to use streamlit
#  16-Aug-2022 

# Thank you for studying my code. You will not learn anything new here. 
# The actual embedding is done using parts of the demo code for SPECTER, the rest is original mess by Lucie


import pandas as pd
import streamlit as st
import numpy as np
from typing import Dict, List
import json
import csv
from csv import DictReader
import requests
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder


try:
	print("Beware: Re-running the whole script now!!!!!!!!!!!!!!!!!!")
	if lets_start_embedding:
		print("Blablablablalbal")
except:
    exit
 

URL = "https://model-apis.semanticscholar.org/specter/v1/invoke"
MAX_BATCH_SIZE = 16


def chunks(lst, chunk_size=MAX_BATCH_SIZE):
    """Splits a longer list to respect batch size"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]
        

def embed(papers):
    print("Function embed has been just called")
    print(papers)
    print("\n\nNOW IT MAY TAKE A WHILE TO RUN")
    embeddings_by_paper_id: Dict[str, List[float]] = {}

    for chunk in chunks(papers):
        # Allow Python requests to convert the data above to JSON
        response = requests.post(URL, json=chunk)

        if response.status_code != 200:
            raise RuntimeError("Sorry, something went wrong, please try later!")

        for paper in response.json()["preds"]:
            embeddings_by_paper_id[paper["paper_id"]] = paper["embedding"]

    return embeddings_by_paper_id  
    
    

def create_embeddings(filename):
	   
	st.write(filename)
	myPrefix = filename[:-4] + "_"
		
	with open(filename, 'r', encoding='utf-8-sig') as read_obj:
		# pass the file object to DictReader() to get the DictReader object
		dict_reader = DictReader(read_obj)
		# get a list of dictionaries from dct_reader
		MY_PAPERS = list(dict_reader)
		# print list of dict i.e. rows
		#print(MY_PAPERS)
    
		all_embeddings = embed(MY_PAPERS)

	print(all_embeddings)
	all_embeddings=pd.DataFrame(all_embeddings)
	all_embeddings.to_csv(myPrefix + 'embeddings.csv')
	print(all_embeddings.head())
	#save a version with titles as index
	df = pd.read_csv(filename)
	print(df.head())
	all_embeddings=all_embeddings.T
	print(all_embeddings.head())
	all_embeddings.index=df.index
	print(all_embeddings.head())
	all_embeddings['title']=df['title']
	print(all_embeddings.head())
	all_embeddings.set_index('title', inplace=True)
	print(all_embeddings.head())
	all_embeddings.to_csv(myPrefix + 'embeddings.T_with_title.csv')
	
	

def prepare_input(filename, df, unique_ID_col,title_col,abstract_col,additional_cols_list):
    try:
           
        myPrefix = filename[:-4] + "_"
        
        myColumns = [unique_ID_col] + [title_col] + [abstract_col] + additional_cols_list
        df=df[myColumns]
           
		# PERFORM CLEANING NOW:
	    		
        #let's drop rows with empty DOIs
        #df.dropna(subset=['DOI'], axis = 0, inplace=True)
        df.dropna(subset=[unique_ID_col], axis = 0, inplace=True)
        df.reset_index(drop=True, inplace = True)
        #df.set_index('paper_id', inplace=True)
        #print(df.tail())
        #it turns out occassionally WoS file can contain DUPLICATES (the row with the same DOI is there twice...weird but happened to me in AdvSci)
        #so let's remove duplicates:
        #df = df.drop_duplicates('DOI', keep='last')
        df = df.drop_duplicates(unique_ID_col, keep='last')
        df.reset_index(drop=True, inplace = True)
        df.set_index(unique_ID_col, inplace=True)
        print('After dropping duplicates:')
        print(df.tail())
    
        #### OK LET'S SAVE IT AS REFERENCE FILE
        
        df.to_csv(myPrefix + 'index_reference.csv')
        
        #####NOW WE DROP ALL THE "ADDITIONAL COLUMNS:
        
        df = df.drop(columns = additional_cols_list, axis=1)
        print(df.head())
        print(df.columns)
        
        #### NOW WE RENAME THE COLUMNS FOR SPECTER:
        df.index.names = ['paper_id']
        df=df.rename(columns={title_col:'title', 'Abstract':'abstract'})  #'DOI':'paper_id',  --- I have already did that above, it can't be done by renaming columns because DOI was an index
        print(df.head())
    
        df.to_csv(myPrefix + 'input4specter.csv')
        print("Done - your input file and reference file are ready.")
        st.write("Done - your input file and reference file are ready.")
        
        #lets_start_embedding = st.button("Generate embeddings!")
        #if lets_start_embedding:
            
            
            #tmp = myPrefix + 'input4specter.csv'
            #st.write(f"Creating embeddings from {tmp}") 
            #create_embeddings(tmp)
        
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        #tk.messagebox.showerror("Error :)", "There was an error!\nHowever, the world will continue to exist. Probably.")
        message = template.format(type(ex).__name__, ex.args)
        st.error(message, icon="🚨")
    
   
        exit


	
st.title("Create embeddings for short documents using SPECTER")


"This app will first prepare an input file for the public SPECTER API from your source csv file **such as search results saved from Web of Science or a report from ScholarOne**."
"Your source file can be any csv file with at least 3 columns as follows:"
" - some sort of a unique ID (DOI, ms ID etc)"
" - document title"
" - document abstract"
"You can create embeddings from titles only and leave the abstract column empty."
"However, for further work with the embeddings, it is not recommended to have a mixed dataset with some embeddings created only from title and other embeddings from both title and abstract.\
Please check this and admend/filter your file before using it as a source file."
 

"The newly created file to be used as input for the SPECTER API will be called **xxx_input4specter.csv** where **xxx** is the name of your original source file."
"A second file with the columns that you specify below will also be generated and its index will be aligned with the xxx_input4specter.csv file to facilitate \
easy concatenation of the resulting file with the embeddings with additional information for your downstream analyses."

"**Note that all records with missing identifiers (typically DOI) will be removed and duplicate rows (by ID) will also be removed.**"

"That is why the index of the resulting input4specter file may be misaligned with the original source file, which is the reason why in addition we are genereting the reference file." 

uploaded_file = st.file_uploader("Select your csv file with documents for which you wish to create embeddings:")

if uploaded_file is not None:
  #print(uploaded_file.name)
  dataframe = pd.read_csv(uploaded_file)
  gb = GridOptionsBuilder.from_dataframe(dataframe, min_column_width=100)
  AgGrid(dataframe.head(5), gridOptions=gb.build(),fit_columns_on_grid_load=True)
  
  all_columns = dataframe.columns
  #all_columns = all_columns.tolist()
  remaining_columns =  all_columns   
  
  unique_ID_column = st.selectbox('Select which column should be used as your unique ID column', (all_columns))
  #st.write('You selected:', unique_ID_column)
    
  #remaining_columns.remove(unique_ID_column)               
  #removing doesn't work because the whole block is run at once, I'd of course need to find a way to control the flow and display the selection
  #boxes only after the previous selection has been made, so I gues I'd have to nest some if blocks or something like that. For now I'll do it as simple as possible
  
  title_column = st.selectbox('Select which column holds the document title', (remaining_columns))
  
  #remaining_columns.remove(abstract_column)
  
  abstract_column = st.selectbox('Select which column holds the document abstract', (remaining_columns))
  
  #remaining_columns.remove(abstract_column)
  
  "Select any additional columns you wish to include in the reference file. Typically you may want to select the publication year, document type, journal, citations etc. in your downstream analysis."
  additional_columns = st.multiselect( 'Select additional columns for your reference file:', (remaining_columns))
  #st.write('You selected:', additional_columns)
  
  "Make sure you have selected the correct columns and press the button to prepare your files:"
   
  buttoning = st.button('PREPARE FILES')
  
  if buttoning:
    prepare_input(uploaded_file.name,dataframe,unique_ID_column,title_column,abstract_column,additional_columns)
	
	# if I simply put the embedding button after the prepare input function so I can only click it if I have "prepared" files
	# it will appear only then , but the trouble is that as soon as I click the embedding button, the WHOLE script starts running again
	# which means that in that moment I click the button, it disappears and the function won't run
    # For that reason I'm going to make use of the session_state which holds the state the whole time the session is running  -
    # I am going to add the "my_prefix_string" field to the session_state now when the prepare_input function has been run - 
    # this should work for distinguishing if the function has been run or not, unless the function crashes, then it;s going to be screwed up.   

    if ('my_prefix_string' not in st.session_state):
      st.session_state['my_prefix_string'] = uploaded_file.name[:-4] + "_"
	  #now I only display embedding button after preparing the files
	  #lets_start_embedding = st.button("Generate embeddings!")
	  
#display embedding button only if prepare_input function has been run:
if 'my_prefix_string' in st.session_state:
    lets_start_embedding = st.button("Generate embeddings!")      



    if lets_start_embedding:
       #st.write("BLABLA")
       tmp = st.session_state['my_prefix_string'] + 'input4specter.csv'
       st.write(f"Creating embeddings from {tmp}") 
       create_embeddings(tmp)
