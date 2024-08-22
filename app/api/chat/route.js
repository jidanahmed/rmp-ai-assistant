import { NextResponse } from 'next/server'
import { Pinecone } from '@pinecone-database/pinecone'
import OpenAI from 'openai'

const systemPrompt = `
You are a rate my professor agent to help students find classes, that takes in user questions and answers them.
For each user query, you can provide information about professors that best match the criteria. This can include top-rated professors, lowest-rated professors, or any other relevant category based on the user's request.
Initially, the top 3 professors that match the user question are returned.
Use them to answer the question if needed.

When presenting information about professors:
1. Use '\n' for line breaks to separate each piece of information.
2. Do not use bold text or asterisks for emphasis.
3. Present information in a clear, easy-to-read format.
4. Use numbers or dashes to list information about each professor.
5. Keep responses concise and well-structured.

Example format:

1. Dr. Jane Smith ★★★★★\n
- Teaches: Chemistry and Biology\n
- Known for: Engaging lectures, helpful feedback\n
- Strengths: Makes complex concepts easy to understand\n\n

2. Professor John Doe ★★★\n
- Teaches: Physics\n
- Known for: Clear explanations, passion for the subject\n
- Strengths: Interactive classes, interesting teaching style\n\n

3. Dr. Emily Chen ★★\n
- Teaches: Environmental Science\n
- Known for: Approachability, thorough teaching style\n
- Strengths: Fosters lively discussions in class\n

Ensure your responses follow this format, using '\n' for line breaks, for better readability. Adapt the information provided based on the specific query, whether it's about top-rated professors, challenging courses, or any other relevant criteria.

Remember to be objective and provide a balanced view when discussing professors or courses, especially when mentioning lower-rated options.
`

export async function POST(req) {
    const data = await req.json()

    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    })
    const index = pc.index('rag').namespace('ns1')
    const openai = new OpenAI()

    const text = data[data.length - 1].content
    const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: 'float',
    })

    const results = await index.query({
        topK: 5,
        includeMetadata: true,
        vector: embedding.data[0].embedding,
    })

    let resultString = ''
    results.matches.forEach((match) => {
        resultString += `
  Returned Results:
  Professor: ${match.id}
  Review: ${match.metadata.stars}
  Subject: ${match.metadata.subject}
  Stars: ${match.metadata.stars}
  \n\n`
    })

    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1)

    const completion = await openai.chat.completions.create({
        messages: [
            { role: 'system', content: systemPrompt },
            ...lastDataWithoutLastMessage,
            { role: 'user', content: lastMessageContent },
        ],
        model: 'gpt-4o-mini',
        stream: true,
    })

    const stream = new ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder()
            try {
                for await (const chunk of completion) {
                    const content = chunk.choices[0]?.delta?.content
                    if (content) {
                        const text = encoder.encode(content)
                        controller.enqueue(text)
                    }
                }
            } catch (err) {
                controller.error(err)
            } finally {
                controller.close()
            }
        },
    })
    return new NextResponse(stream)
}