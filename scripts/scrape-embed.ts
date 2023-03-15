import { Document } from 'langchain/document';
import { PDFLoader } from 'langchain/document_loaders';
// import * as fs from 'fs/promises';
import fs from 'fs';
import { CustomWebLoader } from '@/utils/custom_web_loader';
import type { SupabaseClient } from '@supabase/supabase-js';
import { Embeddings, OpenAIEmbeddings } from 'langchain/embeddings';
import { SupabaseVectorStore } from 'langchain/vectorstores';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { supabaseClient } from '@/utils/supabase-client';
import { pdfPaths } from '@/config/urls';

import path from 'path';
import axios from 'axios';
import { basename } from 'path';

async function downloadPDFsFromURLs(pdfUrls: string[]): Promise<string[]> {
  const dir = './documents';
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir);
  }

  const pdfPaths: string[] = [];

  for (const pdfUrl of pdfUrls) {
    const response = await axios.get(pdfUrl, { responseType: 'arraybuffer' });
    const pdfBuffer = response.data;
    const pdfName = basename(pdfUrl);
    const pdfPath = `${dir}/${pdfName}`;

    fs.writeFileSync(pdfPath, pdfBuffer);
    console.log(`Downloaded ${pdfUrl} to ${pdfPath}`);
    pdfPaths.push(pdfPath);
  }

  console.log(`All PDFs downloaded to ${dir}`);
  return pdfPaths;
}




async function extractDataFromPDFPaths(pdfPaths: string[]): Promise<Document[]> {
  console.log('extracting data from PDFs...');
  const localPDFs: string[] = [];
  const remotePDFs: string[] = [];

  for (const pdfPath of pdfPaths) {
    if (path.isAbsolute(pdfPath) || pdfPath.startsWith('http://') || pdfPath.startsWith('https://')) {
      remotePDFs.push(pdfPath);
    } else {
      localPDFs.push(pdfPath);
    }
  }

  const localDocs = await extractDataFromPDFs(localPDFs);
  const remoteDocs = await extractDataFromRemotePDFs(remotePDFs);
  const documents = [...localDocs, ...remoteDocs];

  console.log('data extracted from PDFs');
  const json = JSON.stringify(documents);
  // await fs.writeFile('frankpdfs.json', json);
  console.log('json file containing data saved on disk');
  return documents;
}

async function extractDataFromPDFs(pdfPaths: string[]): Promise<Document[]> {
  const documents: Document[] = [];

  for (const pdfPath of pdfPaths) {
    const docs = await extractDataFromPDF(pdfPath);
    documents.push(...docs);
  }

  return documents;
}

async function extractDataFromRemotePDFs(pdfUrls: string[]): Promise<Document[]> {
  const documents: Document[] = [];

  for (const pdfUrl of pdfUrls) {
    const response = await axios.get(pdfUrl, { responseType: 'arraybuffer' });
    const pdfBuffer = response.data;
    const docs = await extractDataFromPDF(pdfBuffer);
    documents.push(...docs);
  }

  return documents;
}


async function extractDataFromPDF(pdfPath: string): Promise<Document[]> { 
  try {
    const loader = new PDFLoader(pdfPath);
    const docs = await loader.load();
    return docs;
  } catch (error) {
    console.error(`Error while extracting data from ${pdfPath}: ${error}`);
    return [];
  }
}

// async function extractDataFromPDFs(pdfPaths: string[]): Promise<Document[]> {
//   console.log('extracting data from PDFs...');
//   const documents: Document[] = [];
//   for (const pdfPath of pdfPaths) {

//     // if pdfPath is a url, download it first
//     // if (pdfPath.startsWith('http')) {


//     const docs = await extractDataFromPDF(pdfPath);
//     documents.push(...docs);
//   }
//   console.log('data extracted from PDFs');
//   const json = JSON.stringify(documents);
//   // await fs.writeFile('frankpdfs.json', json);
//   console.log('json file containing data saved on disk');
//   return documents;
// }

async function embedDocuments(
  client: SupabaseClient,
  docs: Document[],
  embeddings: Embeddings,
) {
  console.log('creating embeddings...');
  await SupabaseVectorStore.fromDocuments(client, docs, embeddings);
  console.log('embeddings successfully stored in supabase');
}

async function splitDocsIntoChunks(docs: Document[]): Promise<Document[]> {
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 1,
  });
  return await textSplitter.splitDocuments(docs);
}

(async function run(pdfPaths: string[]) {
  try {
    const localPaths = await downloadPDFsFromURLs(pdfPaths);
    // Load data from each PDF
    const rawDocs = await extractDataFromPDFPaths(localPaths);
    // Split docs into chunks for OpenAI context window
    const docs = await splitDocsIntoChunks(rawDocs);
    // wait 5 seconds
    

    // for (const doc of docs) {
      // console.log('waiting 5 seconds before embedding next document...')
      // await new Promise((resolve) => setTimeout(resolve, 5000));
      // console.log('embedding document:');
      // try {
        await embedDocuments(supabaseClient, docs, new OpenAIEmbeddings());
        
      // } catch (error) {
      //   console.log('error occured:', error);
      // }
      
      
    // }
    // Embed docs into Supabase
    
  } catch (error) {
    console.log('error occured:', error);
  }
})(pdfPaths);
