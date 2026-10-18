import requests
import streamlit as st

st.set_page_config(page_title='Enterprise Knowledge Assistant', page_icon='📚', layout='wide')
st.title('Enterprise GenAI Knowledge Assistant')
st.caption('Ask questions over your documents and inspect the retrieved evidence.')
api_url = st.sidebar.text_input('API URL', 'http://127.0.0.1:8000')

uploaded = st.file_uploader('Upload documents', type=['pdf', 'docx', 'txt', 'md', 'csv'], accept_multiple_files=True)
if st.button('Index documents', type='primary', disabled=not uploaded):
    with st.spinner('Extracting, chunking and indexing...'):
        try:
            payload = [('files', (f.name, f.getvalue(), f.type or 'application/octet-stream')) for f in uploaded]
            response = requests.post(f"{api_url.rstrip('/')}/ingest", files=payload, timeout=180)
            if response.ok:
                st.success('Indexing complete')
                st.json(response.json())
            else:
                st.error(f'Ingestion failed ({response.status_code}): {response.text}')
        except requests.RequestException as exc:
            st.error(f'Could not reach API: {exc}')

st.divider()
question = st.text_area('Ask a question', placeholder='What does the policy say about annual leave?')
top_k = st.slider('Evidence chunks', 1, 10, 5)
if st.button('Ask', type='primary', disabled=len(question.strip()) < 3):
    with st.spinner('Retrieving evidence and generating an answer...'):
        try:
            response = requests.post(f"{api_url.rstrip('/')}/query", json={'question': question, 'top_k': top_k}, timeout=120)
            if response.ok:
                result = response.json()
                st.subheader('Answer')
                st.write(result['answer'])
                st.subheader('Retrieved evidence')
                for source in result.get('sources', []):
                    label = f"[{source['citation']}] {source['source']}"
                    if source.get('page'):
                        label += f" — page {source['page']}"
                    if source.get('row'):
                        label += f" — row {source['row']}"
                    with st.expander(label):
                        st.caption(f"Vector distance: {source.get('distance')}")
                        st.write(source['excerpt'])
            else:
                st.error(f'Query failed ({response.status_code}): {response.text}')
        except requests.RequestException as exc:
            st.error(f'Could not reach API: {exc}')

with st.sidebar:
    st.divider()
    if st.button('Check API health'):
        try:
            st.json(requests.get(f"{api_url.rstrip('/')}/health", timeout=5).json())
        except requests.RequestException as exc:
            st.error(str(exc))
    if st.button('View metrics'):
        try:
            st.json(requests.get(f"{api_url.rstrip('/')}/metrics", timeout=5).json())
        except requests.RequestException as exc:
            st.error(str(exc))
