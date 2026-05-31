from collections import Counter
import numpy as np
from proyecto_recuperacion import process_raw, inverse_index, process_corpus, download_dataset, tf_idf_similarity
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

def jaccard_inverted(query_processed: str, df_corpus: pd.DataFrame, inv_index: dict):
    query_terms = set(query_processed.split())
    N = len(df_corpus)
    scores = np.zeros(N)

    # por cada documento, contar cuántos términos de la query tiene
    intersection = np.zeros(N)
    for term in query_terms:
        if term in inv_index['docs']:
            for doc_id in inv_index['docs'][term]:
                intersection[doc_id] += 1

    # términos únicos por documento
    doc_unique_terms = df_corpus['tokenized'].apply(lambda t: len(set(t))).values

    # |union| = |A| + |B| - |intersección|
    query_size = len(query_terms)
    union = doc_unique_terms + query_size - intersection

    scores = np.divide(intersection, union, out=np.zeros(N, dtype=float), where=union != 0)
    ranking = scores.argsort()[::-1]
    return ranking, scores

def tfidf_inverted(query_processed: str, df_corpus: pd.DataFrame, inv_index: dict):
    query_terms = query_processed.split()
    N = len(df_corpus)

    # precomputar norma completa de cada documento
    doc_norms = np.zeros(N)
    for term, doc_freqs in inv_index['docs'].items():
        df_t = len(doc_freqs)
        idf = np.log(N / df_t)
        for doc_id, tf in doc_freqs.items():
            w = (1 + np.log(tf)) * idf
            doc_norms[doc_id] += w ** 2
    doc_norms = np.sqrt(doc_norms)

    # dot product solo con términos de la query
    scores = np.zeros(N)
    query_norm_sq = 0.0

    for term in query_terms:
        if term not in inv_index['docs']:
            continue
        doc_freqs = inv_index['docs'][term]
        df_t = len(doc_freqs)
        idf = np.log(N / df_t)

        q_weight = (1 + np.log(query_terms.count(term))) * idf
        query_norm_sq += q_weight ** 2

        for doc_id, tf in doc_freqs.items():
            d_weight = (1 + np.log(tf)) * idf
            scores[doc_id] += d_weight * q_weight

    query_norm = np.sqrt(query_norm_sq)

    # normalizar con coseno
    norm = doc_norms * query_norm
    scores = np.divide(scores, norm, out=np.zeros(N, dtype=float), where=norm != 0)

    ranking = scores.argsort()[::-1]
    return ranking, scores

def bm25_scores(query_processed: str, df_corpus: pd.DataFrame, inv_index: dict, k1: float = 1.5, b: float = 0.75):
    query_terms = query_processed.split()
    N = len(df_corpus)

    doc_lengths = df_corpus['tokenized'].apply(len).values
    avgdl = doc_lengths.mean()

    scores = np.zeros(N)

    for term in query_terms:
        if term not in inv_index['docs']:
            continue

        doc_freqs = inv_index['docs'][term]
        n_t = len(doc_freqs)

        # IDF de BM25
        idf = np.log((N - n_t + 0.5) / (n_t + 0.5) + 1)

        for doc_id, tf in doc_freqs.items():
            dl = doc_lengths[doc_id]
            # saturación de TF + penalización por longitud
            numerador = tf * (k1 + 1)
            denominador = tf + k1 * (1 - b + b * (dl / avgdl))
            scores[doc_id] += idf * (numerador / denominador)

    ranking = scores.argsort()[::-1]
    return ranking, scores

from sentence_transformers import SentenceTransformer
import faiss

def build_embeddings(df_corpus: pd.DataFrame):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generando embeddings...")
    doc_embeddings = model.encode(df_corpus['raw'].tolist(), show_progress_bar=True)
    
    dimension = doc_embeddings.shape[1] 
    index = faiss.IndexFlatIP(dimension) 
    
    faiss.normalize_L2(doc_embeddings)
    index.add(doc_embeddings.astype('float32'))
    
    return model, index

def semantic_search(query_text: str, model, faiss_index, df_corpus: pd.DataFrame, top_n: int = 5):
    query_embedding = model.encode([query_text])
    faiss.normalize_L2(query_embedding)
    
    scores, indices = faiss_index.search(query_embedding.astype('float32'), top_n)
    
    ranking = indices.flatten()
    scores = scores.flatten()
    return ranking, scores

def show_results(query_text, tokens, ranking, scores, df_corpus, model_name, top_n=5):
    print(f"{'='*80}")
    print(f"  Modelo: {model_name}")
    print(f"  Query: {query_text}")
    print(f"  Tokens procesados: {tokens}")
    print(f"{'='*80}")
    print(f"  {'Rank':<6} {'Doc ID':<10} {'Score':<12} {'Título'}")
    print(f"  {'-'*70}")
    for i, idx in enumerate(ranking[:top_n]):
        print(f"  {i+1:<6} {df_corpus.loc[idx, 'doc_id']:<10} {scores[idx]:<12.4f} {df_corpus.loc[idx, 'title']}")
    print()


def main():
    print("Cargando dataset...")
    path = download_dataset()
    df_corpus = process_corpus(path + "/ModApte_train.csv")
    
    print("Construyendo índices...")
    inv_index = inverse_index(df_corpus)

    model, faiss_index  = build_embeddings(df_corpus)
    
    modelos = {
        '1': 'Jaccard',
        '2': 'TF-IDF Coseno',
        '3': 'BM25',
        '4': 'Embeddings Semánticos',
        '5': 'Todos'
    }
    
    print("\nSistema de Recuperación de Información")
    print("Escriba 'salir' para terminar.\n")
    
    while True:
        query_text = input("Ingrese su consulta: ").strip()
        if query_text.lower() == 'salir':
            print("Hasta luego!")
            break
        if not query_text:
            print("La consulta no puede estar vacía.\n")
            continue
        
        print("\nModelos disponibles:")
        for key, nombre in modelos.items():
            print(f"  {key}. {nombre}")
        
        opcion = input("Seleccione modelo: ").strip()
        if opcion not in modelos:
            print("Opción no válida.\n")
            continue
        
        try:
            processed = process_raw(query_text)
            
            if opcion in ('1', '5'):
                ranking, scores = jaccard_inverted(' '.join(processed), df_corpus, inv_index)
                show_results(query_text, processed, ranking, scores, df_corpus, "Jaccard")
            
            if opcion in ('2', '5'):
                ranking, scores = tfidf_inverted(' '.join(processed), df_corpus, inv_index)
                show_results(query_text, processed, ranking, scores, df_corpus, "TF-IDF Coseno")
            
            if opcion in ('3', '5'):
                ranking, scores = bm25_scores(' '.join(processed), df_corpus, inv_index)
                show_results(query_text, processed, ranking, scores, df_corpus, "BM25")
            
            if opcion in ('4', '5'):
                ranking, scores = semantic_search(' '.join(processed), model, faiss_index, df_corpus)
                show_results(query_text, processed, ranking, scores, df_corpus, "Embeddings Semánticos")
        
        except Exception as e:
            print(f"Error procesando consulta: {e}\n")

if __name__ == "__main__":
    main()