import streamlit as st
st.title('teste')

n1 = st.number_input('peso:')
n2 = st.number_input('altura:',value = 0.1)

imc = n1/(n2**2)
if st.button('calcular IMC'):
    if imc:
        st.success(imc)
#------------------------------------------------

#3

#formulario

st.caption('CADASTRO SIMPLES')

nome = st.text_input('Nome:')
idade = st.number_input('Idade:')
email = st.text_input('E-mail:')
altura = st.number_input('Altura:')

if st.button('Cadastrar'):
    st.success('Pessoa cadastrada')
#------------------------------------------------

#4
#tabuada
numero =  st.number_input('numero: ')


if st.button('Calcular:'):
    for x in range(0,11):
        calculo = x * numero
        st.write(x , 'x', numero, '=', calculo)
        st.write(f'{x} X {numero} = {calculo}')