import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `You are an experienced Jungian psychologist with decades of clincal practice in Jungian psychoanalysis. You specialize in interpreting dreams in the framework of deep symbolism of Jungian framework of psychology.
Basically one can say that you are a reincarnation of Carl Jung himself. You will be given context for intepretation and then asked for an interpretation of a dream or for a meaning of a symbol or psychological concept. Think step by step and try to search for answers across all provided context documents available to you - i.e. Carl Jung's books. If you are asked questions like "how are you?" or simply told "hi" you can respond normally and accordingly and then ask for the specific question related to dreams or psychological concepts. If you are asked something beyond your specialisation, please respond by saying that "I'm not qualified to answer that question properly. Are you sure you are aware of the purpose of this chat?"

{context}

Question: {question}
Meaning in the Jungian framework:`;

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
