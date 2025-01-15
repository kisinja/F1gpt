import { OpenAI } from "@langchain/openai";
import { OpenAI as openAIEmbeddings } from "openai";
import { OpenAIStream, StreamingTextResponse } from 'ai';
import { DataAPIClient } from "@datastax/astra-db-ts";

const {
    OPENAI_API_KEY,
    ASTRA_DB_NAMESPACE,
    ASTRA_DB_COLLECTION,
    ASTRA_DB_API_ENDPOINT,
    ASTRA_DB_TOKEN
} = process.env;

const openai = new OpenAI({
    apiKey: OPENAI_API_KEY,
});

const openEmbedding = new openAIEmbeddings();

const client = new DataAPIClient(ASTRA_DB_TOKEN);

const db = client.db(ASTRA_DB_API_ENDPOINT, {
    namespace: ASTRA_DB_NAMESPACE
});

export async function POST(req: Request) {
    try {
        const { messages } = await req.json();
        const latestMessage = messages[messages?.length - 1]?.content;

        let docContext = "";

        const embedding = await openEmbedding.embeddings.create({
            model: "text-embedding-3-small",
            input: latestMessage,
            encoding_format: "float"
        });

        try {
            const collection = db.collection(ASTRA_DB_COLLECTION);
            const cursor = collection.find(null, {
                sort: {
                    $vector: embedding.data[0].embedding,
                },
                limit: 10
            });

            const documents = await cursor.toArray();
            const docsMap = documents?.map(doc => doc.text);

            docContext = JSON.stringify(docsMap);
        } catch (error) {
            console.error(error);
            docContext = "";
        }

        const template = {
            role: "system",
            content: `
                You are an AI assistant who knows everything about Formula One. Use the below context to augment what you know about Formula One racing. The context will provide you with the most recent page data from wikipedia, the official F1 website and others. If the context doesn't include the info you need answer based on your existing knowledge and don't mention the source of your info or what the context does or doesn't include. Format responses using markdown where applicable and don't return images.

                -----------------------
                START CONTEXT
                ${docContext}
                END CONTEXT
                -----------------------
                QUESTION: ${latestMessage}
                -----------------------
            `
        };

        const response = await openEmbedding.chat.completions.create({
            model: "gpt-4",
            stream: true,
            messages: [template, ...messages]
        });

        const stream = OpenAIStream(response);

        return new StreamingTextResponse(stream);

    } catch (e) {
        throw e;
    }
}