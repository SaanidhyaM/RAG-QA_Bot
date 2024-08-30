import os
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from pinecone import Pinecone, ServerlessSpec
import tensorflow as tf

# Load the text file containing business data
with open('data.txt', 'r') as f:
    data = f.read()

# Initialize the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('t5-base')
model = TFAutoModelForSeq2SeqLM.from_pretrained('t5-base')

# Initialize the Pinecone client with the new method
try:
    pc = Pinecone(api_key='key')
except Exception as e:
    print(f"Error initializing Pinecone client: {e}")
    exit(1)

# Use a valid index name according to Pinecone's naming convention
index_name = 'business-qa-index'
try:
    # Check if the index exists and delete it if necessary
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)
        print(f"Index '{index_name}' deleted.")
except Exception as e:
    print(f"Error deleting Pinecone index: {e}")
    exit(1)

try:
    # Check if the index already exists before creating it
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
except Exception as e:
    print(f"Error creating Pinecone index: {e}")
    exit(1)

# Add the text data to the index
vectors = []
for i, text in enumerate(data.split('\n')):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors='tf'
    )
    
    decoder_input_ids = tf.convert_to_tensor([[tokenizer.pad_token_id]])
    
    outputs = model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        decoder_input_ids=decoder_input_ids,
        return_dict=True,
        output_hidden_states=True
    )

    encoder_hidden_states = outputs.encoder_hidden_states[-1]
    vector = encoder_hidden_states[:, 0, :].numpy().flatten().tolist()
    vectors.append((str(i), vector, {'text': text}))

try:
    # Connect to the index
    index = pc.Index(index_name)
    # Use the correct method to add vectors
    index.upsert(vectors)
    print(f"Successfully added vectors to index '{index_name}'.")
except Exception as e:
    print(f"Error adding vectors to Pinecone index: {e}")
    exit(1)

# RAG system
def get_context(message, max_tokens=3000, min_score=0.3, get_only_text=True):
    # Get the embeddings of the input message
    inputs = tokenizer.encode_plus(
        message,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors='tf'
    )
    decoder_input_ids = tf.convert_to_tensor([[tokenizer.pad_token_id]])
    
    outputs = model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        decoder_input_ids=decoder_input_ids,
        return_dict=True,
        output_hidden_states=True
    )
    embedding = outputs.encoder_hidden_states[-1][:, 0, :].numpy().flatten().tolist()

    try:
        # Connect to the index
        index = pc.Index(index_name)
        # Query the index using keyword arguments
        results = index.query(vector=embedding, top_k=10)
        print("Query results:", results)
        
        # Filter the results based on the minimum score
        filtered_results = [result for result in results.matches if result.score >= min_score]
        print("Filtered results:", filtered_results)

        # Check if metadata is not None
        for result in filtered_results:
            print("Metadata:", result.metadata)

        # Return the context as a string or a set of ScoredVectors
        if get_only_text:
            context = ' '.join([result.metadata.get('text', '') if result.metadata else '' for result in filtered_results])
        else:
            context = filtered_results
        return context
    except Exception as e:
        print(f"Error querying Pinecone index: {e}")
        exit(1)

def conversation_chain(messages):
    context = ''
    for message in messages:
        context += get_context(message)
    return {'answer': context}

# Example queries
query1 = "What is the company's mission?"
result1 = conversation_chain([query1])
print("Result for query 1:", result1['answer'])

query2 = "What are the company's values?"
result2 = conversation_chain([query2])
print("Result for query 2:", result2['answer'])

query3 = "What is the company's history?"
result3 = conversation_chain([query3])
print("Result for query 3:", result3['answer'])
