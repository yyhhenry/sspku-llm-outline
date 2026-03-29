from concurrent.futures import ThreadPoolExecutor, as_completed

import tokenizers
import typer
from lab_1806_vec_db import VecDB
from openai import OpenAI

global_vec_db = VecDB("data/vector_db")


def steam_messages(client: OpenAI, model_name: str, messages: list[dict]):
    response = client.chat.completions.create(
        messages=messages,
        model=model_name,
        stream=True,
    )
    for chunk in response:
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if isinstance(content, str):
            yield content


def get_embedding(client: OpenAI, model_name: str, text: str) -> list[float]:
    response = client.embeddings.create(
        input=text,
        model=model_name,
    )
    return response.data[0].embedding


def handle_pdf_file(file_path: str):
    from PyPDF2 import PdfReader

    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def handle_txt_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def split_text_into_chunks(
    text: str, chunk_size: int = 1000, overlap: int = 100
) -> list[str]:
    assert chunk_size > overlap, "chunk_size must be greater than overlap"
    tokenizer: tokenizers.Tokenizer = tokenizers.Tokenizer.from_file(
        "assets/tokenizer.json"
    )

    tokens: list[int] = tokenizer.encode(text).ids

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_text: str = tokenizer.decode(tokens[start:end])
        chunks.append(chunk_text)
        start += chunk_size - overlap

    return chunks


def get_collection_files(collection_files_glob: str) -> list[str]:
    import glob

    files = glob.glob(collection_files_glob)
    if not files:
        raise ValueError(f"No files found for glob: {collection_files_glob}")
    return files


def setup_collection(
    client: OpenAI,
    embed_model_name: str,
    collection_name: str,
    collection_files_glob: str,
    force_recreate_collection: bool,
):
    files = get_collection_files(collection_files_glob)
    print(f"Found {len(files)} files for collection '{collection_name}'")
    print(f"Files: {repr(files)}")

    if collection_name in global_vec_db.get_all_keys():
        if force_recreate_collection:
            print(f"Re-creating collection: {collection_name}")
            global_vec_db.delete_table(collection_name)
        else:
            print(f"Collection already exists: {collection_name}")
            return

    chunk_info_list: list[dict[str, str]] = []

    for file_path in files:
        if file_path.lower().endswith(".pdf"):
            text = handle_pdf_file(file_path)
        elif file_path.lower().endswith(".txt") or file_path.lower().endswith(".md"):
            text = handle_txt_file(file_path)
        else:
            print(f"Unsupported file type: {file_path}")
            continue

        chunks = split_text_into_chunks(text)
        print(f"Split file {file_path} into {len(chunks)} chunks")

        chunk_info_list.extend(
            {
                "source": file_path,
                "index": str(index),
                "text": chunk,
            }
            for index, chunk in enumerate(chunks)
        )

    print(f"Total chunks to process: {len(chunk_info_list)}")

    def process_and_store_chunk(
        chunk_info: dict[str, str],
    ):
        info_line = f"Source: {chunk_info['source']} (Index: {chunk_info['index']})"
        embedding = get_embedding(
            client,
            embed_model_name,
            f"{info_line}\n\n{chunk_info['text']}",
        )
        global_vec_db.create_table_if_not_exists(collection_name, dim=len(embedding))
        global_vec_db.add(
            collection_name,
            embedding,
            chunk_info,
        )
        return f"Processed chunk {chunk_info['index']} from {chunk_info['source']}"

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(
                process_and_store_chunk,
                chunk_info,
            )
            for chunk_info in chunk_info_list
        ]
        for index, future in enumerate(as_completed(futures)):
            result = future.result()
            print(f"[{index + 1}/{len(chunk_info_list)}] {result}")

    print("Collection setup complete.")


def preview_string(s: str, length: int = 60) -> str:
    if len(s) <= length:
        return repr(s)
    else:
        start_length = length // 2
        end_length = length - start_length - 3
        return repr(s[:start_length] + "..." + s[-end_length:])


def rag_query(
    client: OpenAI,
    model_name: str,
    embed_model_name: str,
    collection_name: str,
    query: str,
):
    embedding_instruct = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    embeds = get_embedding(
        client,
        embed_model_name,
        f"Instruct: {embedding_instruct}\nQuery: {query}",
    )

    relevant_docs = global_vec_db.search(
        collection_name,
        embeds,
        k=5,
    )
    prompt_parts: list[str] = []
    for doc, distance in relevant_docs:
        info_line = f"Source: {doc['source']} (Index: {doc['index']}, Similarity: {1 - distance})\n"
        prompt_parts.append(info_line)
        prompt_parts.append(doc["text"])
        prompt_parts.append("\n---\n")
        print(f"Retrieved doc | {info_line.strip()} | {preview_string(doc['text'])}")

    prompt_parts.append(
        f"Answer the following query based on the above passages:\n{query}"
    )
    prompt = "\n".join(prompt_parts)

    messages = [
        {"role": "user", "content": prompt},
    ]
    result_parts: list[str] = []
    for chunk in steam_messages(client, model_name, messages):
        print(chunk, end="", flush=True)
        result_parts.append(chunk)
    print()
    return "".join(result_parts)


def cli(
    api_key: str = typer.Option(
        ...,
        envvar="OPENROUTER_API_KEY",
        help="OpenRouter API key",
    ),
    base_url: str = typer.Option(
        "https://openrouter.ai/api/v1",
        help="OpenAI-compatible API base URL",
    ),
    model_name: str = typer.Option(
        "deepseek/deepseek-v3.2",
        help="Chat model name",
    ),
    embed_model_name: str = typer.Option(
        "qwen/qwen3-embedding-8b",
        help="Embedding model name",
    ),
    collection_name: str = typer.Option(
        "llm-lab-naive-rag",
        help="Vector DB collection name",
    ),
    collection_files_glob: str = typer.Option(
        "assets/*.pdf",
        help="Glob pattern for collection source files",
    ),
    force_recreate_collection: bool = typer.Option(
        False,
        help="Force re-creating the collection even if it exists",
    ),
    interactive: bool = typer.Option(
        False,
        help="Run in interactive query mode",
    ),
):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    setup_collection(
        client,
        embed_model_name,
        collection_name,
        collection_files_glob,
        force_recreate_collection,
    )

    if not interactive:
        example_query = "介绍 MiMo-V2-Flash 使用 MTP 的情况"
        print(f"Example query: {example_query}")
        rag_query(client, model_name, embed_model_name, collection_name, example_query)
        return

    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        print("Answer:")
        rag_query(client, model_name, embed_model_name, collection_name, query)


def main():
    typer.run(cli)


if __name__ == "__main__":
    main()
