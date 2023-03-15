import { supabaseClient } from '@/utils/supabase-client';
import { SupabaseVectorStore } from 'langchain/vectorstores';
import { OpenAIEmbeddings } from 'langchain/embeddings';
import { VectorDBQAChain } from 'langchain/chains';
import { openai } from '@/utils/openai-client';

const query = 'Waht was the impact of Woody debris emanating from Uawa Forest?';

const model = openai;

async function searchForDocs() {
  const vectorStore = await SupabaseVectorStore.fromExistingIndex(
    supabaseClient,
    new OpenAIEmbeddings(),
  );

  /*uncomment below to test similarity search */
    const results = await vectorStore.similaritySearch(query, 20);

    console.log("results", results);

  const chain = VectorDBQAChain.fromLLM(model, vectorStore);

  //Ask a question
  const response = await chain.call({
    query: query,
  });

  console.log('response', response);
}

(async () => {
  await searchForDocs();
})();
