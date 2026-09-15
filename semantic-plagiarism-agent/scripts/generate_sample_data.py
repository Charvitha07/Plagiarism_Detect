import os
import docx
import base64

def generate_samples():
    os.makedirs("sample_data", exist_ok=True)
    
    with open("sample_data/english.txt", "w", encoding="utf-8") as f:
        f.write("Artificial intelligence is transforming the world rapidly. It is a very powerful technology. Many industries are adopting it to automate tasks.")
        
    with open("sample_data/paraphrase.txt", "w", encoding="utf-8") as f:
        f.write("The world is being changed at a rapid pace by AI. This technology is incredibly potent. Numerous business sectors are integrating it to handle repetitive work.")
        
    with open("sample_data/hindi.txt", "w", encoding="utf-8") as f:
        f.write("कृत्रिम बुद्धिमत्ता दुनिया को तेजी से बदल रही है। यह एक बहुत ही शक्तिशाली तकनीक है। कई उद्योग कार्यों को स्वचालित करने के लिए इसे अपना रहे हैं।")
        
    with open("sample_data/telugu.txt", "w", encoding="utf-8") as f:
        f.write("కృత్రిమ మేధస్సు ప్రపంచాన్ని వేగంగా మారుస్తోంది. ఇది చాలా శక్తివంతమైన సాంకేతికత. పనులను ఆటోమేట్ చేయడానికి అనేక పరిశ్రమలు దీనిని స్వీకరిస్తున్నాయి.")
        
    with open("sample_data/unrelated.txt", "w", encoding="utf-8") as f:
        f.write("Quantum computing relies on the principles of quantum mechanics. It uses qubits instead of classical bits.")
        
    doc = docx.Document()
    doc.add_paragraph("Artificial intelligence is transforming the world rapidly. It is a very powerful technology. Many industries are adopting it to automate tasks.")
    doc.save("sample_data/document.docx")
    
    # Valid Minimal Base64 Encoded PDF
    pdf_b64 = "JVBERi0xLjQKMSAwIG9iago8PCAvVHlwZSAvQ2F0YWxvZyAvUGFnZXMgMiAwIFIgPj4KZW5kb2JqCjIgMCBvYmoKPDwgL1R5cGUgL1BhZ2VzIC9LaWRzIFszIDAgUl0gL0NvdW50IDEgPj4KZW5kb2JqCjMgMCBvYmoKPDwgL1R5cGUgL1BhZ2UgL1BhcmVudCAyIDAgUiAvTWVkaWFCb3ggWzAgMCA2MTIgNzkyXSAvUmVzb3VyY2VzIDw8IC9Gb250IDw8IC9GMSA0IDAgUiA+PiA+PiAvQ29udGVudHMgNSAwIFIgPj4KZW5kb2JqCjQgMCBvYmoKPDwgL1R5cGUgL0ZvbnQgL1N1YnR5cGUgL1R5cGUxIC9CYXNlRm9udCAvSGVsdmV0aWNhID4+CmVuZG9iago1IDAgb2JqCjw8IC9MZW5ndGggNDQgPj4Kc3RyZWFtCkJUCi9GMSAxMiBUZgoxMDAgNzAwIFRkCihIZWxsbyBXb3JsZCkgVGoKRVQKZW5kc3RyZWFtCmVuZG9iagp4cmVmCjAgNgowMDAwMDAwMDAwIDY1NTM1IGYgCjAwMDAwMDAwMDkgMDAwMDAgbiAKMDAwMDAwMDA1OCAwMDAwMCBuIAowMDAwMDAwMTE1IDAwMDAwIG4gCjAwMDAwMDAyMjMgMDAwMDAgbiAKMDAwMDAwMDMxMSAwMDAwMCBuIAp0cmFpbGVyCjw8IC9TaXplIDYgL1Jvb3QgMSAwIFIgPj4Kc3RhcnR4cmVmCjQwNgolJUVPRgo="
    with open("sample_data/dummy.pdf", "wb") as f:
        f.write(base64.b64decode(pdf_b64))

if __name__ == "__main__":
    generate_samples()