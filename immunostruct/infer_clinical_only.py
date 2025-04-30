import os
import argparse
import torch
from dgl.dataloading import GraphDataLoader

from data import collate, SplitDataset, ClinicalDataset
from models.mapping import model_map
from utils import seed_everything, update_paths
from procedures import inference_clinical_only

if __name__ == "__main__":
    print('STARTING')
    print('REMEMBER TO: export LD_LIBRARY_PATH=/vast/palmer/pi/krishnaswamy_smita/jcr222/conda/conda_envs/immuno/lib:$LD_LIBRARY_PATH')

    parser = argparse.ArgumentParser(description="Entry point.")
    # Model parameters
    parser.add_argument("--model-dir", default="$ROOT/results/")
    parser.add_argument("--model-filename", default="HybridModel_Comparative-wtds_False-lr_pt_0.001-lr_ft_0.0001-cc_0.01-ssl_False-ep_40-bs_128-fseq_True-seql_False-fs_23-cs_3-seed_1_finetune.pt")
    parser.add_argument("--model", default="HybridModel_Comparative", type=str)
    parser.add_argument("--use-wt-for-downstream", action='store_true')
    parser.add_argument("--gcn-layers", default=5, type=int)
    parser.add_argument("--vae-hidden-dim", default=512, type=int)
    parser.add_argument("--vae-latent-dim", default=32, type=int)
    parser.add_argument("--gat-hidden-channels", default=64, type=int)
    parser.add_argument("--property-embedding-dim", default=8, type=int)
    parser.add_argument("--self-supervision", action="store_true")
    
    # Dataset parameters
    parser.add_argument("--feature-size", default=23, type=int)
    parser.add_argument("--coord-size", default=3, type=int)
    parser.add_argument("--full-sequence", action="store_true")

    # Training parameters
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=1, type=int)
    
    # Data paths
    parser.add_argument("--graph-dir", default="$ROOT/data/graph_pyg/", type=str)
    parser.add_argument("--seq-path", default="$ROOT/data/hadrup_cancer_df_29K.txt", type=str)

    # Save paths
    parser.add_argument("--figure-save-dir", default="$ROOT/figures/ImmunoCancer/", type=str)

    config = parser.parse_args()

    #Update paths
    update_paths(config)

    # Model save paths.
    model_path = os.path.join(config.model_dir, config.model_filename)
    print(f'SAVED MODEL PATH: {model_path}')

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    seed_everything(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    
    print('Loading Model')
    # Define model (adjust parameters as needed)
    input_dim = 283 * 21 if config.full_sequence else 11 * 21
    model = model_map[config.model](
        vae_input_dim=input_dim,
        device=device,
        use_wt_for_downstream=config.use_wt_for_downstream,
        gcn_layers=config.gcn_layers,
        vae_hidden_dim=config.vae_hidden_dim,
        vae_latent_dim=config.vae_latent_dim,
        gat_hidden_channels=config.gat_hidden_channels,
        property_embedding_dim=config.property_embedding_dim
    )

    model.load_trained(model_path, new_head=False)
    model.to(device)

    print('Retriving clinical dataset')

    # Defining the Inference Dataset
    clinical_dataset = ClinicalDataset(config,
                                       graph_directory=config.graph_dir,
                                       seq_path=config.seq_path)


    clinical_dataset = SplitDataset(clinical_dataset, "inference", binary=True, full=config.full_sequence, 
                                    comparative=True, 
                                    return_amino_acid=False)
                                    
    clinical_loader = GraphDataLoader(clinical_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=False, num_workers=config.num_workers)


    print('running inference')

    test_stats = inference_clinical_only(config, 
                                        model, 
                                        device,
                                        clinical_loader=clinical_loader,
                                        fig_save_folder=os.path.join(config.figure_save_dir, "results"))

    print('DONE')
