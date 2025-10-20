import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

def calculate_similarity(smiles1, smiles2):
    """Calculate Morgan fingerprint similarity between two SMILES strings"""
    try:
        # Convert SMILES to RDKit molecules
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        # Check if conversion was successful
        if mol1 is None or mol2 is None:
            return None
        
        # Generate Morgan fingerprints (ECFP4)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        
        # Calculate Tanimoto similarity
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        return similarity
    
    except Exception as e:
        print(f"Error calculating similarity for {smiles1} and {smiles2}: {e}")
        return None

def main():
    # Load the CSV file
    try:
        df = pd.read_csv('real_data.csv')
        print(f"Loaded {len(df)} rows from real_data.csv")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    # Check if required columns exist
    required_columns = ["LIMO compound", "Closest REAL compound"]
    if not all(col in df.columns for col in required_columns):
        print(f"CSV must contain columns: {required_columns}")
        return
    
    # Calculate similarity for each row
    print("Calculating similarities...")
    similarities = []
    
    for idx, row in df.iterrows():
        limo_compound = row["LIMO compound"]
        closest_real_compound = row["Closest REAL compound"]
        similarity = calculate_similarity(limo_compound, closest_real_compound)
        similarities.append(similarity)
        
        # Print progress every 100 rows
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} / {len(df)} rows")
    
    # Add similarity column to the dataframe
    df["Similarity"] = similarities
    
    # Calculate average similarity (excluding None values)
    valid_similarities = [s for s in similarities if s is not None]
    if valid_similarities:
        average_similarity = sum(valid_similarities) / len(valid_similarities)
        print(f"Average RDKit similarity: {average_similarity:.4f}")
    else:
        print("No valid similarities calculated")
    
    # Save the updated dataframe to a new CSV
    output_file = 'real_data_with_similarity.csv'
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
