from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate

model = OllamaLLM(
    model="gemma:2b",
)

template = """
          You are an expert in ansering questions about a pizza restaurant.
          
          Here are some relevant reviews: {reviews}
          
          Here is the question to answer: {question}
          
          
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    reviews = chain.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)