import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def initialise_vectorstore(pdf, progress=gr.Progress()):
    progress(0, desc="Reading PDF")

    loader = PyPDFLoader(pdf.name)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)

    progress(0.5, desc="Initialising Vectorstore")

    vectorstore = Chroma.from_documents(
        splits,
        embedding=HuggingFaceEmbeddings()
    )

    progress(1, desc="Complete")

    return vectorstore, progress

def initialise_chain(llm, vectorstore, progress=gr.Progress()):

    progress(0, desc="Initialising LLM")

    llm = HuggingFaceEndpoint(
        repo_id=llm,
        task="text-generation",
        max_new_tokens=512,
        top_k=4,
        temperature=0.1
    )

    chat = ChatHuggingFace(
        llm=llm, 
        verbose=True
    )

    progress(0.5, desc="Initialising RAG Chain")

    retriever = vectorstore.as_retriever()

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(chat, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    progress(0.9, desc="Complete")

    return rag_chain, progress

def send(message, rag_chain, chat_history):
    response = rag_chain.invoke({"input": message})
    chat_history.append((message, response["answer"]))

    return "", chat_history

with gr.Blocks() as demo:

    vectorstore = gr.State()
    rag_chain = gr.State()
    
    gr.Markdown("<H1>Talk to Documents</H1>")
    gr.Markdown("<H3>Upload and ask questions about your PDF files</H3>")
    gr.Markdown("<H6>Note: This project uses LangChain to perform RAG (Retrieval Augmented Generation) on PDF files, allowing users to ask any questions related to their contents. When a PDF file is uploaded, it is embedded and stored in an in-memory Chroma vectorstore, which the chatbot uses as a source of knowledge when aswering user questions.</H6>")

    with gr.Tab("Vectorstore"):
        with gr.Row():
            input_pdf = gr.File()
        with gr.Row():
            with gr.Column(scale=1, min_width=0):
                pass
            with gr.Column(scale=2, min_width=0):
                initialise_vectorstore_btn = gr.Button(
                    "Initialise Vectorstore",
                    variant='primary'
                )
            with gr.Column(scale=1, min_width=0):
                pass
        with gr.Row():
            vectorstore_initialisation_progress = gr.Textbox(value="None", label="Initialization")

    with gr.Tab("RAG Chain"):
        with gr.Row():
            language_model = gr.Radio(["microsoft/Phi-3-mini-4k-instruct", "mistralai/Mistral-7B-Instruct-v0.2", "HuggingFaceH4/zephyr-7b-beta", "mistralai/Mixtral-8x7B-Instruct-v0.1"])
        with gr.Row():
            with gr.Column(scale=1, min_width=0):
                pass
            with gr.Column(scale=2, min_width=0):
                initialise_chain_btn = gr.Button(
                    "Initialise RAG Chain",
                    variant='primary'
                )
            with gr.Column(scale=1, min_width=0):
                pass
        with gr.Row():
            chain_initialisation_progress = gr.Textbox(value="None", label="Initialization")

    with gr.Tab("Chatbot"):
        with gr.Row():
            chatbot = gr.Chatbot()
        with gr.Row():
            message = gr.Textbox()

    initialise_vectorstore_btn.click(fn=initialise_vectorstore, inputs=input_pdf, outputs=[vectorstore, vectorstore_initialisation_progress])
    initialise_chain_btn.click(fn=initialise_chain, inputs=[language_model, vectorstore], outputs=[rag_chain, chain_initialisation_progress])
    message.submit(fn=send, inputs=[message, rag_chain, chatbot], outputs=[message, chatbot])

demo.launch()