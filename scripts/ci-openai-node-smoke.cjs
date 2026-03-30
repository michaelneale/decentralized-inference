#!/usr/bin/env node
/* Official openai-node smoke against a mesh-llm OpenAI-compatible endpoint. */

const OpenAI = require('openai');

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--base-url') {
      args.baseUrl = argv[++i];
    } else {
      throw new Error(`unknown argument: ${arg}`);
    }
  }
  if (!args.baseUrl) {
    throw new Error('--base-url is required');
  }
  return args;
}

async function streamedText(stream) {
  const parts = [];
  let sawChoice = false;

  for await (const chunk of stream) {
    const choices = chunk.choices || [];
    for (const choice of choices) {
      sawChoice = true;
      const content = choice?.delta?.content;
      if (typeof content === 'string' && content.length > 0) {
        parts.push(content);
      }
    }
  }

  if (!sawChoice) {
    throw new Error('stream returned no choices');
  }

  const text = parts.join('').trim();
  if (!text) {
    throw new Error('stream returned no content');
  }
  return text;
}

async function main() {
  const args = parseArgs(process.argv);
  const client = new OpenAI({
    apiKey: 'mesh-llm-ci',
    baseURL: args.baseUrl,
  });

  const models = await client.models.list();
  if (!models.data.length) {
    throw new Error('models.list returned no models');
  }
  const model = models.data[0].id;
  console.log(`Using model: ${model}`);

  const response = await client.chat.completions.create({
    model,
    messages: [{ role: 'user', content: 'Say hello in exactly 4 words.' }],
    max_tokens: 32,
    temperature: 0,
  });

  const message = response.choices?.[0]?.message?.content?.trim();
  if (!message) {
    throw new Error('non-streaming chat returned empty content');
  }
  console.log(`Non-streaming response: ${message}`);

  const stream = await client.chat.completions.create({
    model,
    messages: [{ role: 'user', content: 'Count from one to three.' }],
    max_tokens: 32,
    temperature: 0,
    stream: true,
  });

  const text = await streamedText(stream);
  console.log(`Streaming response: ${text}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack || error.message : String(error));
  process.exit(1);
});
