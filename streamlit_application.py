import streamlit as st
import asyncio
import os
from retrieval_pipeline_final_consolidated import retrieve_docs,gpt_4o

llm = gpt_4o

st.set_page_config(page_title="Document Retrieval System", layout="wide")
st.title("📄 MMAHR : Multi-Modal Multi-Agentic Hierarchical RAG")
st.write("Enter a query to retrieve relevant documents and generate responses.")

# User Input
user_query = st.text_area("🔍 Enter your query:")

if st.button("Retrieve Documents"):
    if user_query.strip():
        st.write("⏳ Processing your query...")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(retrieve_docs(user_query, llm))

        st.subheader("📝 Response:")
        st.markdown(response["answer"], unsafe_allow_html=True)

        supported_extensions = {
            "spreadsheet": [".xlsx", ".xlsm", ".xlsb", ".ods"],
            "image": [".png", ".jpg", ".jpeg"],
            "pdf": [".pdf"]
        }

        if response.get("sources"):
            st.subheader("📌 Sources:")

            for source in response["sources"]:
                if any(source.endswith(ext) for ext in supported_extensions["spreadsheet"]):
                    st.markdown(f"📂 **Download Spreadsheet:** [Open here]({source})")

                elif any(source.endswith(ext) for ext in supported_extensions["image"]):
                    if source.startswith("http"):
                        st.image(source, width=400)
                    elif os.path.exists(source):
                        st.image(source, width=400)
                    else:
                        st.warning(f"❗ **Error:** File not found: {source}")

                elif any(source.endswith(ext) for ext in supported_extensions["pdf"]):
                    st.markdown(f"📄 **View PDF:** [Open here]({source})")
                    st.components.v1.iframe(source, width=800, height=400)

                else:
                    st.markdown(f"🔗 **Reference:** [Open here]({source})")

    else:
        st.warning("⚠️ Please enter a query.")
