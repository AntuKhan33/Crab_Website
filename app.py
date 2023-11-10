import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np
from PIL import Image
# Core Pkgs
import streamlit as st 
# Data Viz Pkg
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns 

model = pickle.load(open('model.sav', 'rb'))
model2 = pickle.load(open('model_Size_of_Increment.sav', 'rb'))

st.title('Crab Price Prediction')
st.header('Select All The Value ')
image = Image.open('crab.jpg')
#st.image(image, '')

# FUNCTION
def user_report():
  Carapace_Width_Before_cm = st.slider('Carapace Width Before (cm)', 5.45,9.56, 0.01 )
  Carapace_Width_After_cm = st.slider('Carapace Width After (cm)', 6.06,10.56, .01 )
  Body_Weight_Before_g = st.slider('Body Weight Before (g)', 44.3,139.8, .01 )
  Body_Weight_After_g = st.slider('Body Weight After (g)', 52.8,148.8, .01 )
  


  user_report_data = {
      'Carapace Width Before (cm)':Carapace_Width_Before_cm,
      'Carapace Width After (cm)':Carapace_Width_After_cm,
      'Body Weight Before (g)':Body_Weight_Before_g,
      'Body Weight After (g)':Body_Weight_After_g,
      
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

user_data = user_report()
st.header('Selected Crab Data')
st.write(user_data)





increment = model2.predict(user_data)
st.subheader('Crab Size of increment ')
st.subheader('$'+str(np.round(increment[0], 2)))
salary = model.predict(user_data)
st.subheader('Crab Price')
st.subheader('$'+str(np.round(salary[0], 2)))
def main():
	"""Semi Automated ML App with Streamlit """

	activities = ["EDA","Plots","About"]	
	choice = st.sidebar.selectbox("Select Activities",activities)

	if choice == 'EDA':
		st.subheader("Exploratory Data Analysis")

		data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())

			if st.checkbox("Show Shape"):
				st.write(df.shape)

			if st.checkbox("Show Columns"):
				all_columns = df.columns.to_list()
				st.write(all_columns)

			if st.checkbox("Summary"):
				st.write(df.describe())

			if st.checkbox("Show Selected Columns"):
				selected_columns = st.multiselect("Select Columns",all_columns)
				new_df = df[selected_columns]
				st.dataframe(new_df)

			if st.checkbox("Show Value Counts"):
				st.write(df.iloc[:,-1].value_counts())

			

			if st.checkbox("Correlation Plot(Seaborn)"):
				st.write(sns.heatmap(df.corr(),annot=True))
				st.pyplot()


			if st.checkbox("Pie Plot"):
				all_columns = df.columns.to_list()
				column_to_plot = st.selectbox("Select 1 Column",all_columns)
				pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
				st.write(pie_plot)
				st.pyplot()



	elif choice == 'Plots':
		st.subheader("Data Visualization")
		data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())


			if st.checkbox("Show Value Counts"):
				st.write(df.iloc[:,-1].value_counts().plot(kind='bar'))
				st.pyplot()
		
			# Customizable Plot

			all_columns_names = df.columns.tolist()
			type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
			selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

			if st.button("Generate Plot"):
				st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

				# Plot By Streamlit
				if type_of_plot == 'area':
					cust_data = df[selected_columns_names]
					st.area_chart(cust_data)

				elif type_of_plot == 'bar':
					cust_data = df[selected_columns_names]
					st.bar_chart(cust_data)

				elif type_of_plot == 'line':
					cust_data = df[selected_columns_names]
					st.line_chart(cust_data)

				# Custom Plot 
				elif type_of_plot:
					cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
					st.write(cust_plot)
					st.pyplot()


	elif choice == 'About':
		st.sidebar.header("Website Created by Mursalin Khan")
		
			
			
    
    
    
    


if __name__ == '__main__':
	main()