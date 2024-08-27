import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def initialise_chatbot(pdf, llm, progress=gr.Progress()):
    progress(0, desc="Reading PDF")

    loader = PyPDFLoader(pdf.name)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)

    progress(0.25, desc="Initialising Vectorstore")

    vectorstore = Chroma.from_documents(
        splits,
        embedding=HuggingFaceEmbeddings()
    )

    progress(0.85, desc="Initialising LLM")

    llm = HuggingFaceEndpoint(
        repo_id=llm,
        task="text-generation",
        max_new_tokens=512,
        top_k=4,
        temperature=0.05
    )

    chat = ChatHuggingFace(
        llm=llm, 
        verbose=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use two sentences maximum and keep the "
        "answer concise and to the point."
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

    return rag_chain, "Complete!"

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

    with gr.Row():
        with gr.Column(scale=1):
            input_pdf = gr.File(label="1. Upload PDF")
            language_model = gr.Radio(label="2. Choose LLM", choices=["microsoft/Phi-3-mini-4k-instruct", "mistralai/Mistral-7B-Instruct-v0.2", "HuggingFaceH4/zephyr-7b-beta", "mistralai/Mixtral-8x7B-Instruct-v0.1"])
            initialise_chatbot_btn = gr.Button(value="3. Initialise Chatbot", variant='primary')
            chatbot_initialisation_progress = gr.Textbox(value="Not Started", label="Initialization Progress")

        with gr.Column(scale=4):
            chatbot = gr.Chatbot(scale=1)
            message = gr.Textbox(label="4. Ask questions about your PDF")

    initialise_chatbot_btn.click(
        fn=initialise_chatbot, inputs=[input_pdf, language_model], outputs=[rag_chain, chatbot_initialisation_progress]
    )
    message.submit(fn=send, inputs=[message, rag_chain, chatbot], outputs=[message, chatbot])

demo.launch()