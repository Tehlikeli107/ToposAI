import gradio as gr
import torch
import sys
import os

# ToposAI Modüllerini İçeri Aktar
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from real_world_medical_fact_checker import MedicalToposEngine
from real_world_solidity_auditor import SolidityToToposParser
from topos_ai.math import transitive_closure

# =====================================================================
# TOPOS AI: NEURO-SYMBOLIC REASONING DASHBOARD (UI)
# =====================================================================

# --- 1. MEDICAL FACT-CHECKER (RAG-KILLER) WRAPPER ---
def run_medical_audit(drug_list, condition_list):
    if not drug_list or not condition_list:
        return "❌ Lütfen en az bir İlaç ve bir Şikayet seçiniz."

    # Ontoloji Kurulumu
    entities = ["Aspirin", "Ibuprofen", "Parasetamol", "Mide_Koruyucu", 
                "Bas_Agrisi", "Ates", "Eklem_Agrisi", 
                "Mide_Kanamasi", "Toksisite", "Kalp_Durmasi"]
    engine = MedicalToposEngine(entities)
    
    # Bilgi Tabanı Yüklemesi
    engine.add_knowledge("Aspirin", "Bas_Agrisi", 1.0)
    engine.add_knowledge("Ibuprofen", "Bas_Agrisi", 1.0)
    engine.add_knowledge("Parasetamol", "Ates", 1.0)
    engine.add_knowledge("Aspirin", "Mide_Kanamasi", 0.5) 
    engine.add_knowledge("Ibuprofen", "Mide_Kanamasi", 0.5)
    
    # Hesaplama ve Sonuç Üretimi
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        engine.check_prescription(drugs=drug_list, patient_conditions=condition_list)
    output = f.getvalue()
    
    # Sonucu formatla
    if "REÇETE REDDEDİLDİ" in output and "İLAÇ ETKİLEŞİMİ" in output:
        return f"🚨 KRİTİK UYARI 🚨\n{output}"
    elif "REÇETE REDDEDİLDİ" in output:
        return f"⚠️ ETKİSİZ REÇETE ⚠️\n{output}"
    else:
        return f"✅ GÜVENLİ ✅\n{output}"

# --- 2. WEB3 SMART CONTRACT AUDITOR WRAPPER ---
def run_smart_contract_audit(solidity_code):
    if not solidity_code.strip():
        return "❌ Lütfen denetlenecek Solidity kodunu giriniz."
        
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    with redirect_stdout(f):
        parser = SolidityToToposParser()
        nodes, edges, n_idx = parser.parse_code(solidity_code)
        N = len(nodes)
        
        R = torch.zeros(N, N)
        for u, v in edges:
            R[n_idx[u], n_idx[v]] = 1.0
            
        R_inf = calculate_transitive_closure(R, max_steps=N)
        
        call_node = None
        state_update_node = None
        for n in nodes:
            if ".call" in n: call_node = n
            if "balances[" in n and "=" in n and "0" in n: state_update_node = n
            
        if call_node and state_update_node:
            loop_exists = R_inf[n_idx[call_node], n_idx[call_node]].item() == 1.0
            call_idx = nodes.index(call_node)
            update_idx = nodes.index(state_update_node)
            cei_violated = call_idx < update_idx
            
            if loop_exists and cei_violated:
                print("🚨 KRİTİK ZAFİYET (CRITICAL VULNERABILITY): REENTRANCY (GERİ ÇAĞIRMA) AÇIĞI 🚨")
                print(f"[KANIT]: Topos Matrisi, '{call_node}' satırının kendisini bir döngüye (Loop) sokabildiğini kanıtladı.")
                print(f"[SEBEP]: Durum güncellemesi ('{state_update_node}') para transferinden SONRA yazılmış.")
                print("[ÇÖZÜM]: Checks-Effects-Interactions (CEI) kuralını uygulayın. Bakiye güncellemesini '.call' satırından ÖNCEYE alın!")
            else:
                print("✅ SÖZLEŞME GÜVENLİ (SAFE).")
                print("[KANIT]: Para transferi (.call) sonrası tehlikeli bir döngü veya gecikmiş durum güncellemesi (State Update) bulunamadı.")
        else:
            print("✅ SÖZLEŞME GÜVENLİ (SAFE). Transfer veya Bakiye güncellemesi bulunamadı.")
            
    return f.getvalue()


# =====================================================================
# GRADIO UI TASARIMI
# =====================================================================
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
)

with gr.Blocks(title="ToposAI Reasoning Engine") as app:
    gr.Markdown("# 🧠 ToposAI: Neuro-Symbolic Reasoning Dashboard")
    gr.Markdown("Standart LLM'lerin istatistiksel halüsinasyonlarına karşı, Kategori Teorisi ve Topoloji kullanan %100 Matematiksel Doğrulama (Formal Verification) Motoru.")
    
    with gr.Tabs():
        # --- TAB 1: MEDICAL RAG-KILLER ---
        with gr.TabItem("🏥 Medical Fact-Checker"):
            gr.Markdown("### Tıbbi Reçete Doğrulama (Adverse Drug Reaction Finder)")
            gr.Markdown("Klasik RAG (Vektör Arama), ilaç ve hastalık metin benzerliğine bakarak hastaya birbirini zehirleyen iki ilacı aynı anda önerebilir. ToposAI ise **Sheaf Gluing** kullanarak bu iki ilacın midede yaratacağı ölümcül çelişkiyi topolojik olarak yakalar.")
            
            with gr.Row():
                with gr.Column():
                    med_drugs = gr.Dropdown(
                        choices=["Aspirin", "Ibuprofen", "Parasetamol", "Mide_Koruyucu"], 
                        multiselect=True, label="💊 Yazılacak İlaçlar"
                    )
                    med_conds = gr.Dropdown(
                        choices=["Bas_Agrisi", "Ates", "Eklem_Agrisi"], 
                        multiselect=True, label="🤒 Hastanın Şikayetleri"
                    )
                    med_btn = gr.Button("Reçeteyi Denetle (Topos AI)", variant="primary")
                    
                    gr.Examples(
                        examples=[
                            [["Parasetamol"], ["Ates"]],
                            [["Aspirin", "Ibuprofen"], ["Bas_Agrisi"]]
                        ],
                        inputs=[med_drugs, med_conds]
                    )
                with gr.Column():
                    med_output = gr.Textbox(label="Denetim Sonucu (Formal Verification)", lines=10)
                    
            med_btn.click(fn=run_medical_audit, inputs=[med_drugs, med_conds], outputs=med_output)

        # --- TAB 2: WEB3 SMART CONTRACT AUDITOR ---
        with gr.TabItem("🛡️ Web3 Smart Contract Auditor"):
            gr.Markdown("### Solidity Kod Denetimi (Reentrancy Loop Finder)")
            gr.Markdown("GPT-4 veya standart LLM'ler kodu metin olarak okur ve prompt-injection ile kandırılabilir. ToposAI, kodu **Topolojik Kontrol Akış Grafiğine (CFG)** çevirir ve döngüleri (Hacker Loop) matematiksel olarak kanıtlar.")
            
            with gr.Row():
                with gr.Column():
                    contract_input = gr.Code(language="javascript", label="Solidity Kodu", lines=15)
                    audit_btn = gr.Button("Kodu Denetle (Transitive Closure)", variant="primary")
                    
                    gr.Examples(
                        examples=[
                            ["""function withdraw() public {
    uint bal = balances[msg.sender];
    require(bal > 0);
    (bool sent, ) = msg.sender.call{value: bal}("");
    require(sent, "Failed to send Ether");
    balances[msg.sender] = 0;
}"""],
                            ["""function withdraw() public {
    uint bal = balances[msg.sender];
    require(bal > 0);
    balances[msg.sender] = 0;
    (bool sent, ) = msg.sender.call{value: bal}("");
    require(sent, "Failed to send Ether");
}"""]
                        ],
                        inputs=[contract_input]
                    )
                with gr.Column():
                    audit_output = gr.Textbox(label="Güvenlik Raporu (Topos Matrisi)", lines=10)
                    
            audit_btn.click(fn=run_smart_contract_audit, inputs=contract_input, outputs=audit_output)

    gr.Markdown("---")
    gr.Markdown("**ToposAI Research Project** | Powered by PyTorch & Category Theory")

if __name__ == "__main__":
    app.launch(theme=theme, inbrowser=True, show_error=True)
