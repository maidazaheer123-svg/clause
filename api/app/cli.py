from app.services.chat import RAGService

def main():
    rag_services = RAGService()

    try:
        query = input("Enter your query: ").strip()
        result = rag_services.ask(question=query)
        print(f"AI: {result.answer}")
    except:
        pass
    finally:
        rag_services.close()  # Ensure resources are cleaned up

if __name__ == "__main__":
    main()
