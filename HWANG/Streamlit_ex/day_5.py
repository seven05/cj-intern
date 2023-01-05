import numpy as np
import altair as alt
import pandas as pd
import streamlit as st

st.header('st.write')

# ex1
st.write('hello, *World!* : sunglasses:')

# ex 2

st.write(1234)

# ex 3

df = pd.DataFrame({
    'first column':[1,2,3,4],
    'second column' : [10,20,30,40]
})

st.write(df)

# ex 4

st.write('Below is a DataFrame:', df, 'Above is dataframe')

# ex 5
df2 = pd.DataFrame(
    np.random.randn(200,3),
    columns=['a','b','c']
)

c = alt.Chart(df2).mark_circle().encode(
    x='a', y='b', size = 'c', color='c', tooltip=['a','b','c']
)
st.write(c)