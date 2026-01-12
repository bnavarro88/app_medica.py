import streamlit as st
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

# --- CONFIGURACIÓN DE LA IA (Mantenemos tu arquitectura) ---
class CerebroHibrido(nn.Module):
    def __init__(self):
        super(CerebroHibrido, self).__init__()
        self.fc1 = nn.Linear(1025, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x_quimica, x_biologia):
        # Aseguramos que ambos tengan la misma dimensión antes de unirlos
        x_quimica = x_quimica.view(-1) 
        x_biologia = x_biologia.view(-1)
        
        # Unimos: 1024 (química) + 1 (biología) = 1025
        x = torch.cat((x_quimica, x_biologia), dim=0)
        
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return torch.tensor(list(fp), dtype=torch.float)

# --- BASE DE DATOS DE ANTIBIÓTICOS (Vademécum interno) ---
VADEMECUM = {
    "Amoxicilina": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C",
    "Ciprofloxacino": "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O",
    "Vancomicina": "CC1C(C(C(O1)OC2C(C(C(OC2OC3C4=CC5=C(C=C4)OC6=C(C=C(C=C6)C(C(C(=O)NC(C(=O)NC(C7=CC(=C(C=C7)O)C8=C(C(=CC(=C8)O)OC9C(C(C(C(O9)CO)O)O)O)C(C(=O)NC(C(=O)N5)CC1=CC(=C(O1)Cl)O)NC(=O)C(C1=CC(=C(C=C1)O)O)N)O)C(=O)O)NC(=O)C(C1=CC(=C(C=C1)O)Cl)NC4=O)O)O)O)O)N)O",
    "Azitromicina": "CCC1C(C(C(N(CC(CC(C(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)OC)N(C)C)C)OC3C(C(CC(O3)C)N)O)C)O)C)C)O)(C)O",
    "Gentamicina": "CC(C1CCC(C(O1)OC2C(CC(C(C2O)OC3C(C(C(O3)CO)O)N)N)N)N)NC"
}

# --- INTERFAZ MÉDICA ---
st.set_page_config(page_title="AI Clinical Assistant", page_icon="🏥")
st.title("🏥 Asistente de Prescripción Diana (IA)")
st.markdown("Optimización de tratamiento basada en el fenotipo bacteriano del paciente.")

@st.cache_resource
def load_model():
    model = CerebroHibrido()
    model.load_state_dict(torch.load('modelo_antibioticos_final.pth'))
    model.eval()
    return model

modelo = load_model()

# Selección del Paciente
with st.container():
    st.subheader("📋 Datos de la Analítica")
    col1, col2 = st.columns(2)
    with col1:
        bacteria = st.selectbox("Patógeno detectado:", 
                                ["E. coli", "Staphylococcus aureus", "Klebsiella pneumoniae", "Enterococcus faecalis", "Pseudomonas aeruginosa"])
    with col2:
        # Lógica de Gram automática basada en microbiología real
        gram_pos = ["Staphylococcus aureus", "Enterococcus faecalis"]
        tipo = "POSITIVA" if bacteria in gram_pos else "NEGATIVA"
        st.info(f"Fisiología detectada: Gram {tipo}")

# Botón de Procesamiento
if st.button("⚡ Calcular Tratamiento Óptimo"):
    st.divider()
    biologia = torch.tensor([1.0] if tipo == "POSITIVA" else [0.0])
    
    resultados = []
    for nombre, smiles in VADEMECUM.items():
        fp = smiles_to_fp(smiles)
        if fp is not None:
            with torch.no_grad():
                # Llamada limpia al modelo
                prob = modelo(fp, biologia).item()
                resultados.append({"Antibiótico": nombre, "Eficacia Predicha": prob})
    
    # Crear tabla de resultados
    df = pd.DataFrame(resultados).sort_values(by="Eficacia Predicha", ascending=False)
    
    # Mostrar el "Ganador"
    mejor = df.iloc[0]
    st.success(f"✅ **Tratamiento Recomendado:** {mejor['Antibiótico']} ({mejor['Eficacia Predicha']:.1%})")
    
    # Mostrar tabla comparativa
    st.table(df.style.format({"Eficacia Predicha": "{:.2%}"}))
    

    st.warning("⚠️ **Nota:** Esta es una herramienta experimental basada en IA. Debe ser validada por un microbiólogo clínico.")
