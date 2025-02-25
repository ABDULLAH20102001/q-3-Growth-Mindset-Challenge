# import streamlit as st
# import pandas as pd
# import os
# from io import BytesIO

# st.set_page_config(page_title="Data Sweeper", layout='wide')

# # Custom CSS
# st.markdown(
#     """
#       <style>
#       .stApp{background-color: black; color:white;}
#       </style>
#      """,
#     unsafe_allow_html=True
#    )

# #title and description
# st.title("👿 Datasweeper Sterling Integrator By Abdullah-Saleem.")
# st.write("Transform your files between CSV and Excel formats with ")

# # File uploader
# uploaded_files = st.file_uploader("Upload your file (accepts CSV or Excel):", type=["csv", "xlsx"], accept_multiple_files=True)

# if uploaded_files:
#    for file in uploaded_files:
#        file_ext = os.path.splitext(file.name)[-1].lower()

#        if file_ext == ".csv":
#           df = pd.read_csv(file) 
#        elif file_ext == "xlsx":
#            df = pd.read_excel(file)
#       #  else:
#       #     st.error(f"unsupported file type:")
#       #     continue
#        else:
#         st.error(f"Unsupported file type: {file.name}")
#         continue



       

        


#   #file details
#    st.write("Perview the head of the Dataframe")
#    st.dataframe(df.head())

#   #data cleaning options
#    st.subheader("Data cleaning options")
#    if st.checkbox(f"Cleian data for {file.name}"):
#       col1,col2 = st.columns(2)

#    with col1:
#     if st.button(f"Remove duplicates from the file:{file.name}"):
#       df.drop_duplicates(inplace=True)
#       # Add functionality here (e.g., remove specific columns or rows)    
#       st.write("✔✅ Duplicates removed!")


#       with col2:
#         if st.button(f"fill missing values for {file.name}"):
#            numeric_cols = df.select_dtypes(include=['number']).columns
#            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
#            st.write("✔✅ Missing values have been filled!")

#       st.subheader("🎁 Select Columns to keep")     
#       columns = st.multiselect(f"Choose columns for{file.name}",df.columns, df.columns, default=df.columns)
#       df = df[columns]


# #data visualizaiton
# st.subheader("Data visualization")
# if st.checkbox(f"Data visualization {file.name}"):
#    st.bar_chart(df.select_dtypes(include='number').iloc[:, :2])


#    #conversion Options
#    st.subheader(" conversion Options")
#    conversion_type = st.radio(f"convert{file.name} to:",["CSV", "Excel"], key=file.name)
#    if st.button(f"Convert{file.name}"):
#       buffer = BytesIO()
#       if conversion_type == "CSV":
#          df.to_csv(buffer, index=False)
#          file_name = file.name.replace(file_ext)

#          mime_type = "text/csv"
#       elif conversion_type == "Excel":
#             df.to.to_excel(buffer, index=False)
#             file_name = file.name.replace(file_ext, ".xlsx")
#             mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             buffer.seek(0)

#             st.download_button(
#                label =f"download {file.name} as {conversion_type}",
#                data=buffer,
#                file_name=file_name,
#                mime=mime_type
#             )

# st.success("All files processed successfully!")            











# import streamlit as st
# import pandas as pd
# import os
# from io import BytesIO

# st.set_page_config(page_title="Data Sweeper", layout='wide')

# # Custom CSS
# st.markdown(
#     """
#       <style>
#       .stApp{background-color: black; color:white;}
#       </style>
#      """,
#     unsafe_allow_html=True
# )

# # title and description
# st.title("👿 Datasweeper Sterling Integrator By Abdullah-Saleem.")
# st.write("Transform your files between CSV and Excel formats with ")

# # File uploader
# uploaded_files = st.file_uploader("Upload your file (accepts CSV or Excel):", type=["csv", "xlsx"], accept_multiple_files=True)

# if uploaded_files:
#     for file in uploaded_files:
#         file_ext = os.path.splitext(file.name)[-1].lower()

#         if file_ext == ".csv":
#             df = pd.read_csv(file)
#         elif file_ext == ".xlsx":
#             df = pd.read_excel(file)
#         else:
#             st.error(f"unsupported file type:")
#             continue

#         # file details
#         st.write("Preview the head of the DataFrame")
#         st.dataframe(df.head())

#         # data cleaning options
#         st.subheader("Data cleaning options")
#         if st.checkbox(f"Clean data for {file.name}"):
#             col1, col2 = st.columns(2)

#             with col1:
#                 if st.button(f"Remove duplicates from the file: {file.name}"):
#                     df.drop_duplicates(inplace=True)
#                     st.write("✔✅ Duplicates removed!")

#             with col2:
#                 if st.button(f"Fill massing values for {file.name}"):
#                     numeric_cols = df.select_dtypes(include=['number']).columns
#                     df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
#                     st.write("✔✅ Missing values have been filled!")

#             st.subheader("🎁 Select Columns to keep")
#             columns = st.multiselect(f"Choose columns for {file.name}", df.columns, df.columns, default=df.columns)
#             df = df[columns]

#         # data visualization
#         st.subheader("Data visualization")
#         if st.checkbox(f"Data visualization for {file.name}"):
#             st.bar_chart(df.select_dtypes(include='number').iloc[:, :2])

#         # conversion options
#         st.subheader("Conversion Options")
#         conversion_ = st.radio(f"Convert {file.name} to:", ["CSV", "Excel"], key=file.name)
#         if st.button(f"Convert {file.name}"):
#             buffer = BytesIO()
#             if conversion_ == "CSV":
#                 df.to_csv(buffer, index=False)
#                 file_name = file.name.replace(file_ext, ".csv")
#                 mime_type = "text/csv"
#             elif conversion_ == "Excel":
#                 df.to_excel(buffer, index=False)
#                 file_name = file.name.replace(file_ext, ".xlsx")
#                 mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             buffer.seek(0)

#             st.download_button(
#                 label=f"Download {file.name} as {conversion_}",
#                 data=buffer,
#                 file_name=file_name,
#                 mime=mime_type
#             )

# st.success("All files processed successfully!")








# import streamlit as st
# import pandas as pd
# import os
# from io import BytesIO

# st.set_page_config(page_title="Data Sweeper", layout='wide')

# # Custom CSS
# st.markdown(
#     """
#       <style>
#       .stApp{background-color: black; color:white;}
#       </style>
#      """,
#     unsafe_allow_html=True
# )

# #title and description
# st.title("👿 Datasweeper Sterling Integrator By Abdullah-Saleem.")
# st.write("Transform your files between CSV and Excel formats with ")

# # File uploader
# uploaded_files = st.file_uploader("Upload your file (accepts CSV or Excel):", type=["csv", "xlsx"], accept_multiple_files=True)

# if uploaded_files:
#    for file in uploaded_files:
#        file_ext = os.path.splitext(file.name)[-1].lower()

#        if file_ext == ".csv":
#           df = pd.read_csv(file) 
#        elif file_ext == ".xlsx":
#            df = pd.read_excel(file)
#        else:
#           st.error(f"Unsupported file type: {file.name}")
#           continue

#        #file details
#        st.write("Preview the head of the Dataframe")
#        st.dataframe(df.head())

#        #data cleaning options
#        st.subheader("Data cleaning options")
#        if st.checkbox(f"Clean data for {file.name}"):
#           col1, col2 = st.columns(2)

#           with col1:
#               if st.button(f"Remove duplicates from the file: {file.name}"):
#                   df.drop_duplicates(inplace=True)
#                   st.write("✔✅ Duplicates removed!")

#           with col2:
#               if st.button(f"Fill missing values for {file.name}"):
#                   numeric_cols = df.select_dtypes(include=['number']).columns
#                   df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
#                   st.write("✔✅ Missing values have been filled!")

#           st.subheader("🎁 Select Columns to keep")
#           columns = st.multiselect(f"Choose columns for {file.name}", df.columns, df.columns, default=df.columns)
#           df = df[columns]

#        # Data visualization
#        st.subheader("Data visualization")
#        if st.checkbox(f"Data visualization for {file.name}"):
#            st.bar_chart(df.select_dtypes(include='number').iloc[:, :2])

#        # conversion Options
#        st.subheader("Conversion Options")
#        conversion_type = st.radio(f"Convert {file.name} to:", ["CSV", "Excel"], key=file.name)
#        if st.button(f"Convert {file.name}"):
#            buffer = BytesIO()
#            if conversion_type == "CSV":
#                df.to_csv(buffer, index=False)
#                file_name = file.name.replace(file_ext, ".csv")
#                mime_type = "text/csv"
#            elif conversion_type == "Excel":
#                df.to_excel(buffer, index=False)
#                file_name = file.name.replace(file_ext, ".xlsx")
#                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#            buffer.seek(0)

#            st.download_button(
#                label=f"Download {file.name} as {conversion_type}",
#                data=buffer,
#                file_name=file_name,
#                mime=mime_type
#            )

# st.success("All files processed successfully!")










# import streamlit as st
# import pandas as pd
# import os
# from io import BytesIO

# st.set_page_config(page_title="Data Sweeper", layout='wide')

# # Custom CSS
# st.markdown(
#     """
#       <style>
#       .stApp{background-color: black; color:white;}
#       </style>
#      """,
#     unsafe_allow_html=True
# )

# #title and description
# st.title("👿 Datasweeper Sterling Integrator By Abdullah-Saleem.")
# st.write("Transform your files between CSV and Excel formats with ")

# # File uploader
# uploaded_files = st.file_uploader("Upload your file (accepts CSV or Excel):", type=["csv", "xlsx"], accept_multiple_files=True)

# if uploaded_files:
#    for file in uploaded_files:
#        file_ext = os.path.splitext(file.name)[-1].lower()

#        if file_ext == ".csv":
#           df = pd.read_csv(file) 
#        elif file_ext == ".xlsx":
#            df = pd.read_excel(file)
#        else:
#           st.error(f"Unsupported file type: {file.name}")
#           continue

#        #file details
#        st.write("Preview the head of the Dataframe")
#        st.dataframe(df.head())

#        #data cleaning options
#        st.subheader("Data cleaning options")
#        if st.checkbox(f"Clean data for {file.name}"):
#           col1, col2 = st.columns(2)

#           with col1:
#               if st.button(f"Remove duplicates from the file: {file.name}"):
#                   df.drop_duplicates(inplace=True)
#                   st.write("✔✅ Duplicates removed!")

#           with col2:
#               if st.button(f"Fill missing values for {file.name}"):

#                   numeric_cols = df.select_dtypes(include=['number']).columns
#                   df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
#                   st.write("✔✅ Missing values have been filled!")

#           st.subheader("🎁 Select Columns to keep")
#           columns = st.multiselect(f"Choose columns for {file.name}", options=df.columns, default=df.columns)
#           df = df[columns]

#        # Data visualization
#        st.subheader("Data visualization")
#        if st.checkbox(f"Data visualization for {file.name}"):
#            numeric_columns = df.select_dtypes(include=['number']).columns
#            if len(numeric_columns) >= 2:
#                st.bar_chart(df[numeric_columns].iloc[:, :2])
#            else:
#                st.warning("Not enough numeric columns for visualization.")

#        # conversion Options
#        st.subheader("Conversion Options")
#        conversion_type = st.radio(f"Convert {file.name} to:", ["CSV", "Excel"], key=file.name)
#        if st.button(f"Convert {file.name}"):
#            buffer = BytesIO()
#            if conversion_type == "CSV":
#                df.to_csv(buffer, index=False)
#                file_name = file.name.replace(file_ext, ".csv")
#                mime_type = "text/csv"
#            elif conversion_type == "Excel":
#                df.to_excel(buffer, index=False)
#                file_name = file.name.replace(file_ext, ".xlsx")
#                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#            buffer.seek(0)

#            st.download_button(
#                label=f"Download {file.name} as {conversion_type}",
#                data=buffer,
#                file_name=file_name,
#                mime=mime_type
#            )

# st.success("All files processed successfully!")









# import streamlit as st
# import pandas as pd
# import os
# from io import BytesIO
# import seaborn as sns
# import matplotlib.pyplot as plt
# from pandas_profiling import ProfileReport
# import matplotlib.dates as mdates
# import numpy as np

# # Page configuration
# st.set_page_config(page_title="Data Sweeper", layout='wide')

# # Custom CSS for UI
# st.markdown("""
#       <style>
#       .stApp{background-color: black; color:white;}
#       </style>
#      """, unsafe_allow_html=True)

# # Title and description
# st.title("👿 Datasweeper Sterling Integrator By Abdullah-Saleem.")
# st.write("Transform your files between CSV, Excel, and JSON formats with ease.")

# # File uploader with multiple files support
# uploaded_files = st.file_uploader("Upload your file(s) (accepts CSV, Excel, or JSON):", type=["csv", "xlsx", "json"], accept_multiple_files=True)

# if uploaded_files:
#     for file in uploaded_files:
#         file_ext = os.path.splitext(file.name)[-1].lower()

#         # Handling different file formats
#         if file_ext == ".csv":
#             df = pd.read_csv(file)
#         elif file_ext == ".xlsx":
#             df = pd.read_excel(file)
#         elif file_ext == ".json":
#             df = pd.read_json(file)
#         else:
#             st.error(f"Unsupported file type: {file.name}")
#             continue

#         # File size limit check (example: 10 MB)
#         file_size = len(file.getvalue())
#         if file_size > 10 * 1024 * 1024:
#             st.warning(f"The file {file.name} is quite large, which may affect performance. Please consider using a smaller file.")

#         # Display file preview
#         st.write(f"Preview the head of the DataFrame for {file.name}:")
#         st.dataframe(df.head())

#         # Data Summary (Basic Statistics)
#         st.subheader(f"Basic Statistics for {file.name}")
#         if st.checkbox("Show data summary and statistics"):
#             st.write(df.describe())

#         # Data cleaning options
#         st.subheader(f"Data cleaning options for {file.name}")
#         if st.checkbox(f"Clean data for {file.name}"):
#             col1, col2 = st.columns(2)

#             with col1:
#                 if st.button(f"Remove duplicates from the file: {file.name}"):
#                     df.drop_duplicates(inplace=True)
#                     st.write("✔✅ Duplicates removed!")

#             with col2:
#                 if st.button(f"Fill missing values for {file.name}"):
#                     numeric_cols = df.select_dtypes(include=['number']).columns
#                     df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
#                     st.write("✔✅ Missing values have been filled!")

#             # Column selection
#             st.subheader("🎁 Select Columns to Keep")
#             columns = st.multiselect(f"Choose columns for {file.name}", options=df.columns, default=df.columns.tolist())
#             df = df[columns]

#             # Remove NaN/Null Values
#             if st.button(f"Remove NaN/Null values for {file.name}"):
#                 df.dropna(inplace=True)
#                 st.write("✔✅ NaN/Null values removed!")

#             # Replace outliers (based on z-score)
#             if st.button(f"Replace outliers in {file.name}"):
#                 from scipy import stats
#                 z_scores = stats.zscore(df.select_dtypes(include=['number']).dropna())
#                 abs_z_scores = abs(z_scores)
#                 filtered_entries = (abs_z_scores < 3).all(axis=1)
#                 df = df[filtered_entries]
#                 st.write("✔✅ Outliers removed!")

#         # Data Filtering and Sorting
#         st.subheader(f"Filter and Sort Data for {file.name}")
#         filter_column = st.selectbox(f"Select column to filter {file.name}", df.columns)
#         filter_value = st.text_input(f"Enter value to filter {file.name}")
#         if filter_value:
#             filtered_df = df[df[filter_column].astype(str).str.contains(filter_value, case=False)]
#             st.write(f"Filtered data based on column {filter_column} with value {filter_value}:")
#             st.dataframe(filtered_df)

#         # Sorting functionality
#         sort_column = st.selectbox(f"Select column to sort {file.name}", df.columns)
#         sort_order = st.radio(f"Sort order for {file.name}", ["Ascending", "Descending"])
#         if st.button(f"Sort data by {sort_column}"):
#             sorted_df = df.sort_values(by=sort_column, ascending=(sort_order == "Ascending"))
#             st.write(f"Sorted data by {sort_column} in {sort_order} order:")
#             st.dataframe(sorted_df)

#         # Data visualization options
#         st.subheader(f"Data visualization for {file.name}")
#         if st.checkbox(f"Show data visualization for {file.name}"):
#             chart_type = st.selectbox("Select chart type", ['Bar Chart', 'Line Chart', 'Pie Chart', 'Scatter Plot'])
#             numeric_columns = df.select_dtypes(include=['number']).columns
#             if len(numeric_columns) >= 2:
#                 if chart_type == 'Bar Chart':
#                     st.bar_chart(df[numeric_columns].iloc[:, :2])
#                 elif chart_type == 'Line Chart':
#                     st.line_chart(df[numeric_columns].iloc[:, :2])
#                 elif chart_type == 'Pie Chart':
#                     st.write(df[numeric_columns[0]].value_counts().plot.pie(autopct='%.2f%%'))
#                 elif chart_type == 'Scatter Plot':
#                     st.pyplot(df.plot.scatter(x=numeric_columns[0], y=numeric_columns[1], c='blue'))
#             else:
#                 st.warning("Not enough numeric columns for visualization.")

#         # Generate Pandas Profiling Report
#         st.subheader(f"Generate Profile Report for {file.name}")
#         if st.checkbox(f"Generate profile report for {file.name}"):
#             profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
#             st_profile_report(profile)

#         # Row search functionality
#         st.subheader(f"Search for specific rows in {file.name}")
#         search_query = st.text_input(f"Search rows in {file.name}:")
#         if search_query:
#             filtered_df = df[df.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)]
#             st.write(f"Found {len(filtered_df)} matching rows:")
#             st.dataframe(filtered_df)

#         # Conversion Options
#         st.subheader("Conversion Options")
#         conversion_type = st.radio(f"Convert {file.name} to:", ["CSV", "Excel", "JSON"], key=file.name)
#         if st.button(f"Convert {file.name}"):
#             buffer = BytesIO()
#             if conversion_type == "CSV":
#                 df.to_csv(buffer, index=False)
#                 file_name = file.name.replace(file_ext, ".csv")
#                 mime_type = "text/csv"
#             elif conversion_type == "Excel":
#                 df.to_excel(buffer, index=False)
#                 file_name = file.name.replace(file_ext, ".xlsx")
#                 mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             elif conversion_type == "JSON":
#                 df.to_json(buffer, orient="records", lines=True)
#                 file_name = file.name.replace(file_ext, ".json")
#                 mime_type = "application/json"
#             buffer.seek(0)

#             # Download button for conversion
#             st.download_button(
#                 label=f"Download {file.name} as {conversion_type}",
#                 data=buffer,
#                 file_name=file_name,
#                 mime=mime_type
#             )

# st.success("All files processed successfully!")











# import streamlit as st
# import pandas as pd
# import os
# from io import BytesIO
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# from pydantic_settings import BaseSettings

# # Page configuration
# st.set_page_config(page_title="Data Sweeper", layout='wide')

# # Custom CSS for UI
# st.markdown("""
#       <style>
#       .stApp{background-color: black; color:white;}
#       </style>
#      """, unsafe_allow_html=True)

# # Title and description
# st.title("👿 Datasweeper Sterling Integrator By Abdullah-Saleem.")
# st.write("Transform your files between CSV, Excel, and JSON formats with ease.")

# # File uploader with multiple files support
# uploaded_files = st.file_uploader("Upload your file(s) (accepts CSV, Excel, or JSON):", type=["csv", "xlsx", "json"], accept_multiple_files=True)

# if uploaded_files:
#     for file in uploaded_files:
#         file_ext = os.path.splitext(file.name)[-1].lower()

#         # Handle file formats
#         if file_ext == ".csv":
#             df = pd.read_csv(file)
#         elif file_ext == ".xlsx":
#             df = pd.read_excel(file)
#         elif file_ext == ".json":
#             df = pd.read_json(file)
#         else:
#             st.error(f"Unsupported file type: {file.name}")
#             continue

#         # File size limit check (example: 10 MB)
#         file_size = len(file.getvalue())
#         if file_size > 10 * 1024 * 1024:
#             st.warning(f"The file {file.name} is quite large, which may affect performance. Please consider using a smaller file.")

#         # Display file preview
#         st.write(f"Preview the head of the DataFrame for {file.name}:")
#         st.dataframe(df.head())

#         # Data Summary (Basic Statistics)
#         st.subheader(f"Basic Statistics for {file.name}")
#         if st.checkbox("Show data summary and statistics"):
#             st.write(df.describe())

#         # Data cleaning options
#         st.subheader(f"Data cleaning options for {file.name}")
#         if st.checkbox(f"Clean data for {file.name}"):

#             col1, col2 = st.columns(2)
#             with col1:
#                 if st.button(f"Remove duplicates from {file.name}"):
#                     df.drop_duplicates(inplace=True)
#                     st.write("✔✅ Duplicates removed!")

#             with col2:
#                 if st.button(f"Fill missing values for {file.name}"):
#                     numeric_cols = df.select_dtypes(include=['number']).columns
#                     df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
#                     st.write("✔✅ Missing values filled with mean!")

#             # Column selection
#             st.subheader("🎁 Select Columns to Keep")
#             columns = st.multiselect(f"Choose columns for {file.name}", options=df.columns, default=df.columns.tolist())
#             df = df[columns]

#             # Remove NaN/Null Values
#             if st.button(f"Remove NaN/Null values for {file.name}"):
#                 df.dropna(inplace=True)
#                 st.write("✔✅ NaN/Null values removed!")

#             # Replace outliers (based on z-score)
#             if st.button(f"Replace outliers in {file.name}"):
#                 from scipy import stats
#                 z_scores = stats.zscore(df.select_dtypes(include=['number']).dropna())
#                 abs_z_scores = abs(z_scores)
#                 filtered_entries = (abs_z_scores < 3).all(axis=1)
#                 df = df[filtered_entries]
#                 st.write("✔✅ Outliers removed!")

#         # Data Filtering and Sorting
#         st.subheader(f"Filter and Sort Data for {file.name}")
#         filter_column = st.selectbox(f"Select column to filter {file.name}", df.columns)
#         filter_value = st.text_input(f"Enter value to filter {file.name}")
#         if filter_value:
#             filtered_df = df[df[filter_column].astype(str).str.contains(filter_value, case=False)]
#             st.write(f"Filtered data based on column {filter_column} with value {filter_value}:")
#             st.dataframe(filtered_df)

#         # Sorting functionality
#         sort_column = st.selectbox(f"Select column to sort {file.name}", df.columns)
#         sort_order = st.radio(f"Sort order for {file.name}", ["Ascending", "Descending"])
#         if st.button(f"Sort data by {sort_column}"):
#             sorted_df = df.sort_values(by=sort_column, ascending=(sort_order == "Ascending"))
#             st.write(f"Sorted data by {sort_column} in {sort_order} order:")
#             st.dataframe(sorted_df)

#         # Data visualization options
#         st.subheader(f"Data visualization for {file.name}")
#         if st.checkbox(f"Show data visualization for {file.name}"):

#             chart_type = st.selectbox("Select chart type", ['Bar Chart', 'Line Chart', 'Pie Chart', 'Scatter Plot'])
#             numeric_columns = df.select_dtypes(include=['number']).columns
#             if len(numeric_columns) >= 2:
#                 if chart_type == 'Bar Chart':
#                     st.bar_chart(df[numeric_columns].iloc[:, :2])
#                 elif chart_type == 'Line Chart':
#                     st.line_chart(df[numeric_columns].iloc[:, :2])
#                 elif chart_type == 'Pie Chart':
#                     st.write(df[numeric_columns[0]].value_counts().plot.pie(autopct='%.2f%%'))
#                 elif chart_type == 'Scatter Plot':
#                     st.pyplot(df.plot.scatter(x=numeric_columns[0], y=numeric_columns[1], c='blue'))
#             else:
#                 st.warning("Not enough numeric columns for visualization.")

#         # Row search functionality
#         st.subheader(f"Search for specific rows in {file.name}")
#         search_query = st.text_input(f"Search rows in {file.name}:")
#         if search_query:
#             filtered_df = df[df.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)]
#             st.write(f"Found {len(filtered_df)} matching rows:")
#             st.dataframe(filtered_df)

#         # Conversion Options
#         st.subheader("Conversion Options")
#         conversion_type = st.radio(f"Convert {file.name} to:", ["CSV", "Excel", "JSON"], key=file.name)
#         if st.button(f"Convert {file.name}"):
#             buffer = BytesIO()
#             if conversion_type == "CSV":
#                 df.to_csv(buffer, index=False)
#                 file_name = file.name.replace(file_ext, ".csv")
#                 mime_type = "text/csv"
#             elif conversion_type == "Excel":
#                 df.to_excel(buffer, index=False)
#                 file_name = file.name.replace(file_ext, ".xlsx")
#                 mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             elif conversion_type == "JSON":
#                 df.to_json(buffer, orient="records", lines=True)
#                 file_name = file.name.replace(file_ext, ".json")
#                 mime_type = "application/json"
#             buffer.seek(0)

#             # Download button for conversion
#             st.download_button(
#                 label=f"Download {file.name} as {conversion_type}",
#                 data=buffer,
#                 file_name=file_name,
#                 mime=mime_type
#             )

# st.success("All files processed successfully!")








# import streamlit as st
# import pandas as pd
# import os
# from io import BytesIO
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# from pydantic_settings import BaseSettings

# # Page configuration
# st.set_page_config(page_title="Data Sweeper", layout='wide')

# # Custom CSS for UI
# st.markdown("""
#       <style>
#       .stApp{background-color: black; color:white;}
#       </style>
#      """, unsafe_allow_html=True)

# # Title and description
# st.title("👿 Datasweeper Sterling Integrator By Abdullah-Saleem.")
# st.write("Transform your files between CSV, Excel, and JSON formats with ease.")

# # File uploader with multiple files support
# uploaded_files = st.file_uploader("Upload your file(s) (accepts CSV, Excel, or JSON):", type=["csv", "xlsx", "json"], accept_multiple_files=True)

# if uploaded_files:
#     for file in uploaded_files:
#         file_ext = os.path.splitext(file.name)[-1].lower()

#         # Handle file formats
#         if file_ext == ".csv":
#             df = pd.read_csv(file)
#         elif file_ext == ".xlsx":
#             df = pd.read_excel(file)
#         elif file_ext == ".json":
#             df = pd.read_json(file)
#         else:
#             st.error(f"Unsupported file type: {file.name}")
#             continue

#         # File size limit check (example: 10 MB)
#         file_size = len(file.getvalue())
#         if file_size > 10 * 1024 * 1024:
#             st.warning(f"The file {file.name} is quite large, which may affect performance. Please consider using a smaller file.")

#         # Display file preview
#         st.write(f"Preview the head of the DataFrame for {file.name}:")
#         st.dataframe(df.head())

#         # Display column options and preview
#         st.subheader(f"Preview and Select Columns from {file.name}")
#         selected_columns = st.multiselect("Select columns to preview", options=df.columns.tolist(), default=df.columns.tolist())
#         st.dataframe(df[selected_columns].head())

#         # Highlight Missing Values
#         st.subheader("Highlight Missing Values")
#         if st.checkbox(f"Highlight missing values for {file.name}"):
#             missing_data = df.isnull().sum()
#             st.write(f"Missing data in each column:")
#             st.write(missing_data)
#             st.write("Highlighted Dataframe (Missing Values marked as `NaN`):")
#             st.dataframe(df.style.applymap(lambda val: 'background-color: yellow' if pd.isna(val) else '', subset=selected_columns))

#         # Data Summary (Basic Statistics)
#         st.subheader(f"Basic Statistics for {file.name}")
#         if st.checkbox("Show data summary and statistics"):
#             st.write(df.describe())

#         # Advanced Customizable Plots
#         st.subheader(f"Advanced Data Visualization for {file.name}")
#         chart_type = st.selectbox("Select chart type", ['Bar Chart', 'Line Chart', 'Pie Chart', 'Scatter Plot'])
#         x_column = st.selectbox(f"Select X-axis column for {file.name}", df.columns)
#         y_column = st.selectbox(f"Select Y-axis column for {file.name}", df.columns)
         
#         if chart_type == 'Bar Chart':
#              st.bar_chart(df[[x_column, y_column]])
#         elif chart_type == 'Line Chart':
#              st.line_chart(df[[x_column, y_column]])
#         elif chart_type == 'Pie Chart':
#              st.write(df[y_column].value_counts().plot.pie(autopct='%1.1f%%'))
#         elif chart_type == 'Scatter Plot':
#              st.pyplot(df.plot.scatter(x=x_column, y=y_column, c='blue'))

#         # Column selection
#         st.subheader("🎁 Select Columns to Keep")
#         columns = st.multiselect(f"Choose columns for {file.name}", options=df.columns, default=df.columns.tolist())
#         df = df[columns]

#         # Remove NaN/Null Values
#         if st.button(f"Remove NaN/Null values for {file.name}"):
#             df.dropna(inplace=True)
#             st.write("✔✅ NaN/Null values removed!")

#         # Replace outliers (based on z-score)
#         if st.button(f"Replace outliers in {file.name}"):
#             from scipy import stats
#             z_scores = stats.zscore(df.select_dtypes(include=['number']).dropna())
#             abs_z_scores = abs(z_scores)
#             filtered_entries = (abs_z_scores < 3).all(axis=1)
#             df = df[filtered_entries]
#             st.write("✔✅ Outliers removed!")

#         # Data Filtering and Sorting
#         st.subheader(f"Filter and Sort Data for {file.name}")
#         filter_column = st.selectbox(f"Select column to filter {file.name}", df.columns)
#         filter_value = st.text_input(f"Enter value to filter {file.name}")
#         if filter_value:
#             filtered_df = df[df[filter_column].astype(str).str.contains(filter_value, case=False)]
#             st.write(f"Filtered data based on column {filter_column} with value {filter_value}:")
#             st.dataframe(filtered_df)

#         # Sorting functionality
#         sort_column = st.selectbox(f"Select column to sort {file.name}", df.columns)
#         sort_order = st.radio(f"Sort order for {file.name}", ["Ascending", "Descending"])
#         if st.button(f"Sort data by {sort_column}"):
#             sorted_df = df.sort_values(by=sort_column, ascending=(sort_order == "Ascending"))
#             st.write(f"Sorted data by {sort_column} in {sort_order} order:")
#             st.dataframe(sorted_df)

#         # Row search functionality
#         st.subheader(f"Search for specific rows in {file.name}")
#         search_query = st.text_input(f"Search rows in {file.name}:")
#         if search_query:
#             filtered_df = df[df.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)]
#             st.write(f"Found {len(filtered_df)} matching rows:")
#             st.dataframe(filtered_df)

#         # Conversion Options
#         st.subheader("Conversion Options")
#         conversion_type = st.radio(f"Convert {file.name} to:", ["CSV", "Excel", "JSON"], key=file.name)
#         if st.button(f"Convert {file.name}"):
#             buffer = BytesIO()
#             if conversion_type == "CSV":
#                 df.to_csv(buffer, index=False)
#                 file_name = file.name.replace(file_ext, ".csv")
#                 mime_type = "text/csv"
#             elif conversion_type == "Excel":
#                 df.to_excel(buffer, index=False)
#                 file_name = file.name.replace(file_ext, ".xlsx")
#                 mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             elif conversion_type == "JSON":
#                 df.to_json(buffer, orient="records", lines=True)
#                 file_name = file.name.replace(file_ext, ".json")
#                 mime_type = "application/json"
#             buffer.seek(0)

#             # Download button for conversion
#             st.download_button(
#                 label=f"Download {file.name} as {conversion_type}",
#                 data=buffer,
#                 file_name=file_name,
#                 mime=mime_type
#             )

#         # Option to download raw, unprocessed data
#         st.subheader(f"Download Raw Data for {file.name}")
#         if st.button(f"Download raw {file.name}"):
#             buffer = BytesIO()
#             df.to_csv(buffer, index=False)
#             buffer.seek(0)
#             st.download_button(
#                 label=f"Download Raw Data as CSV",
#                 data=buffer,
#                 file_name=f"raw_{file.name.replace(file_ext, '.csv')}",
#                 mime="text/csv"
#             )

# st.success("All files processed successfully!")














import streamlit as st
import pandas as pd
import os
from io import BytesIO
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Data Analyzer", layout='wide')

# Custom CSS for UI
st.markdown("""
    <style>
    .stApp{background-color: #2a2a2a; color:white;}
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🔍 Data Analyzer")
st.write("A quick tool to analyze and visualize data. Upload CSV, Excel, or JSON files.")

# File uploader with multiple files support
uploaded_file = st.file_uploader("Upload a file (CSV, Excel, or JSON):", type=["csv", "xlsx", "json"])

if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[-1].lower()

    # Handle file formats
    if file_ext == ".csv":
        df = pd.read_csv(uploaded_file)
    elif file_ext == ".xlsx":
        df = pd.read_excel(uploaded_file)
    elif file_ext == ".json":
        df = pd.read_json(uploaded_file)
    else:
        st.error(f"Unsupported file type: {uploaded_file.name}")
        st.stop()

    # File size limit check (example: 5 MB)
    file_size = len(uploaded_file.getvalue())
    if file_size > 5 * 1024 * 1024:
        st.warning(f"The file {uploaded_file.name} is quite large. It may affect performance.")

    # Display file preview
    st.write("Here’s a preview of your data:")
    st.dataframe(df.head())

    # Column Filtering
    st.subheader("Select Columns for Analysis")
    selected_columns = st.multiselect("Select columns", options=df.columns.tolist(), default=df.columns.tolist())
    st.dataframe(df[selected_columns].head())

    # Handling Missing Data
    st.subheader("Handle Missing Data")
    if st.checkbox("Show missing data count"):
        missing_data = df.isnull().sum()
        st.write("Missing Data Count:")
        st.write(missing_data)

    if st.checkbox("Remove rows with missing data"):
        df_cleaned = df.dropna()
        st.write("Missing values removed.")
        st.dataframe(df_cleaned.head())

    # Basic Statistics
    st.subheader("Data Statistics")
    if st.checkbox("Show statistics summary"):
        st.write(df.describe())

    # Visualization Options
    st.subheader("Visualize the Data")
    chart_type = st.selectbox("Select the type of chart", ['Bar Chart', 'Line Chart', 'Scatter Plot'])
    x_column = st.selectbox("Select X-axis column", df.columns)
    y_column = st.selectbox("Select Y-axis column", df.columns)

    if chart_type == 'Bar Chart':
        st.bar_chart(df[[x_column, y_column]].set_index(x_column))
    elif chart_type == 'Line Chart':
        st.line_chart(df[[x_column, y_column]].set_index(x_column))
    elif chart_type == 'Scatter Plot':
        st.pyplot(df.plot.scatter(x=x_column, y=y_column, c='blue'))

    # Data Sorting
    st.subheader("Sort Data")
    sort_column = st.selectbox("Sort by column", df.columns)
    sort_order = st.radio("Sort order", ["Ascending", "Descending"])
    if st.button("Sort Data"):
        sorted_df = df.sort_values(by=sort_column, ascending=(sort_order == "Ascending"))
        st.write(f"Data sorted by {sort_column} in {sort_order} order:")
        st.dataframe(sorted_df)

    # Convert data to CSV
    st.subheader("Download Data in New Format")
    conversion_type = st.radio("Convert data to", ["CSV", "Excel", "JSON"])
    if st.button(f"Convert to {conversion_type}"):
        buffer = BytesIO()
        if conversion_type == "CSV":
            df.to_csv(buffer, index=False)
            file_name = uploaded_file.name.replace(file_ext, ".csv")
            mime_type = "text/csv"
        elif conversion_type == "Excel":
            df.to_excel(buffer, index=False)
            file_name = uploaded_file.name.replace(file_ext, ".xlsx")
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif conversion_type == "JSON":
            df.to_json(buffer, orient="records", lines=True)
            file_name = uploaded_file.name.replace(file_ext, ".json")
            mime_type = "application/json"
        buffer.seek(0)

        # Download button for conversion
        st.download_button(
            label=f"Download {file_name} as {conversion_type}",
            data=buffer,
            file_name=file_name,
            mime=mime_type
        )
