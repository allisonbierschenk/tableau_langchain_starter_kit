# Deployment Guide

## Environment Variables Required

For the app to work in production, you need to set the following environment variable in your Vercel deployment:

### Required Environment Variables

1. **OPENAI_API_KEY** - Your OpenAI API key for AI functionality
   - Go to your Vercel dashboard
   - Navigate to your project settings
   - Go to "Environment Variables"
   - Add: `OPENAI_API_KEY` with your actual OpenAI API key

## Deployment Steps

1. **Set Environment Variables in Vercel:**
   - Login to Vercel dashboard
   - Select your project
   - Go to Settings → Environment Variables
   - Add `OPENAI_API_KEY` with your OpenAI API key

2. **Deploy the Application:**
   - Push your changes to GitHub
   - Vercel will automatically deploy
   - The app will be available at: `https://tableau-langchain-starter-kit.vercel.app`

3. **Test the Deployment:**
   - Visit the deployed URL
   - Try asking a question like "What data sources do I have access to?"
   - Verify the app works for other users

## Troubleshooting

### Common Issues:

1. **"Failed to fetch" errors:**
   - Check that the frontend is pointing to the correct production URL
   - Verify CORS settings allow your Tableau domain

2. **OpenAI API errors:**
   - Ensure `OPENAI_API_KEY` is set in Vercel environment variables
   - Check that your OpenAI API key is valid and has credits

3. **MCP server connection issues:**
   - The MCP server URL is hardcoded and should work for all users
   - If issues persist, check the MCP server status

## Local Development

For local development, create a `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key_here
```

Then run:
```bash
python web_app.py
```

The app will be available at `http://localhost:8000`
