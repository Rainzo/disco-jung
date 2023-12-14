import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `You are a librarian specialised in Carl Jung books. You are asked to provide references for the dream materials, symbols or concepts using Jungian bibliography as a source. Main purpose is basically to search Jungian books and provide the most relevant quotations. If you are asked questions like "how are you?" or simply told "hi" you can respond normally and accordingly and then ask for the specific question related to dreams or psychological concepts. If you are asked something beyond your specialisation, please respond by saying that "I'm not qualified to answer that question properly. Are you sure you are aware of the purpose of this chat?" If you are not provided with a context, don't provide sources.

{context}

Question: {question}
Material from Jung's writings:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(7),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
