#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from esm_embedder import ESMCEmbedder
from esm2_embedder import ESM2Embedder
from esm1b_embedder import ESM1bEmbedder

def test_embedder(embedder_class, model_name, test_sequences):
    print(f"\nTesting {embedder_class.__name__} with model {model_name}")
    print("=" * 60)
    
    try:
        embedder = embedder_class(model_name=model_name)
        print(f"✓ Model loaded successfully")
        
        embeddings = embedder.embed_sequences(test_sequences, batch_size=2)
        print(f"✓ Generated embeddings for {len(embeddings)} sequences")
        
        if embeddings:
            first_id = list(embeddings.keys())[0]
            embedding_shape = embeddings[first_id].shape
            print(f"✓ Embedding dimension: {embedding_shape[0]}")
            print(f"✓ Expected dimension: {embedder.get_embedding_dim()}")
            
        return True
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

def main():
    test_sequences = [
        ("seq1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        ("seq2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("seq3", "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSMKTVRQERLKSIVRILERSKEPVSGAQ")
    ]
    
    print("Testing Protein Embedding Models")
    print("=" * 60)
    
    # Test ESMC
    success_esmc = test_embedder(ESMCEmbedder, "esmc_300m", test_sequences)
    
    # Test ESM-2
    success_esm2 = test_embedder(ESM2Embedder, "esm2_t6_8M_UR50D", test_sequences)
    
    # Test ESM-1b
    success_esm1b = test_embedder(ESM1bEmbedder, "esm1b_t33_650M_UR50S", test_sequences)
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"ESM-C: {'✓ PASSED' if success_esmc else '✗ FAILED'}")
    print(f"ESM-2: {'✓ PASSED' if success_esm2 else '✗ FAILED'}")
    print(f"ESM-1b: {'✓ PASSED' if success_esm1b else '✗ FAILED'}")
    
    if all([success_esmc, success_esm2, success_esm1b]):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())