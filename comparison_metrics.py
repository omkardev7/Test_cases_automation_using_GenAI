# comparison_metrics.py
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import nltk

# Download NLTK data for BLEU score
nltk.download('punkt')

def calculate_semantic_similarity(correct_answer, generated_answer):
    """
    Calculate semantic similarity between the correct answer and the generated answer.
    """
    # Load a pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode the sentences to get their embeddings
    correct_embedding = model.encode(correct_answer, convert_to_tensor=True)
    generated_embedding = model.encode(generated_answer, convert_to_tensor=True)

    # Compute cosine similarity between the embeddings
    similarity = util.cos_sim(correct_embedding, generated_embedding).item()

    return similarity

def calculate_bleu_score(correct_answer, generated_answer):
    """
    Calculate BLEU score between the correct answer and the generated answer.
    """
    # Tokenize the sentences
    reference = [nltk.word_tokenize(correct_answer)]
    candidate = nltk.word_tokenize(generated_answer)

    # Calculate BLEU score
    bleu_score = sentence_bleu(reference, candidate)
    return bleu_score

def calculate_rouge_score(correct_answer, generated_answer):
    """
    Calculate ROUGE scores between the correct answer and the generated answer.
    """
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Calculate ROUGE scores
    scores = scorer.score(correct_answer, generated_answer)
    return scores