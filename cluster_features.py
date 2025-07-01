import matplotlib.pyplot as plt
import numpy as np
import openslide
from openslide.deepzoom import DeepZoomGenerator
import os
from pathlib import Path
import pandas as pd
import random
import sys
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing

from sklearn.metrics import pairwise_distances, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import euclidean_distances

from rl_benchmarks.constants import PREPROCESSED_DATA_DIR

# cluster features into eg 100 clusters
# show 3 closest samples from each dataset for some pre-defined samples
# Use cosine similarity.

def extract_features(model="iBOTViTBasePANCAN", datatype="all", tuned_dir="phikon_tuned/dinov2_tcga_phikon", use_features=100):
    """
    
    INPUT:
        datatype: str. {lusc, luad, all}
    """
    model_to_type = {"iBOTViTBasePANCAN":"Phikon", "uni":"UNI", "virchow":"Virchow", "virchow2":"Virchow 2", "phikon2":"Phikon 2", "phikon_tuned": "Phikon tuned", "v2_tuned": "Virchow2 tuned", "p2_tuned": "Phikon2 tuned"}
    feature_type = model_to_type[model]
    savename = "".join([d[0] for d in feature_type.lower().split(" ")])
    
    if not "tuned" in model:
        feat_dir = PREPROCESSED_DATA_DIR.joinpath("slides_classification/features_2")
    else:
        tp = "/home/vilde/data/slides_classification/features_tuned/_2/"+tuned_dir
        feat_dir = PREPROCESSED_DATA_DIR.joinpath(tp)
        print("Using phikon tuned model from", tp)
        savename = tuned_dir.split("/")[-1]
    
    print("savename", savename)
    use_label = "OS.time"
    np.random.seed(48)

    use_label_dict = {"TCGA": "ajcc_pathologic_tumor_stage", "S36": "UICC", "UNN": "p_stage", "NLST": "Stage"}
    #TCGA eg Stage IIB. S36 eg IIB. UNN eg Stage IIb. NLST eg 110 -> IA w dict.
    nlst_stage: {110:"IA", 120: "IB", 210: "IIA", 220: "IIB", 310: "IIIA", 320: "IIIB", 400: "IV", 888: "TNM not available", 994: "Carcinoid, cannot be assessed"}
    
    if datatype=="lusc":
        _data = ["TCGA/TCGA_LUSC", "S36_LUSC", "UNN_LUSC"]#, "NLST_LUSC"]
    elif datatype=="luad":
        _data = ["TCGA/TCGA_LUAD", "S36_LUAD", "UNN_LUAD"]#, "NLST_LUAD"]
    elif datatype=="all":
        _data =["TCGA/TCGA_LUSC", "TCGA/TCGA_LUAD", "S36_LUSC", "S36_LUAD", "UNN_LUSC", "UNN_LUAD"]
    else:
        print("Unvalid datatype", datatype)
        sys.exit()

    from string import digits
    remove_digits = str.maketrans('', '', digits)
    features = np.array([])
    labels = []
    stage_labels = []
    domains = {"TCGA":0, "S36":0, "UNN":0, "NLST":0}
    _indexes = []
    slides_id = []
    slides_id_tileindexes = np.array([])

    tiles_counter = 0

    for ld in _data:
        if not "tuned" in model:
            feature_dir = os.path.join(feat_dir, model, ld)
        else: # "tuned" in model:
            feature_dir = os.path.join(feat_dir, ld)
        ld_name = ld.split("_")[0].split("/")[0]
        datatype = ld.split("_")[-1]
        if ld_name == "TCGA":
            ld_labels = str(os.path.join("/home/vilde/code/Phikon/HistoSSLscaling/raw/slides_classification", ld_name, "clinical/survival/survival_labels_"+ld_name.lower()+"_"+datatype.lower()+".csv"))
        else:
            ld_labels = str(os.path.join("/home/vilde/code/Phikon/HistoSSLscaling/raw/slides_classification", ld_name, "clinical/survival_labels_"+ld_name.lower()+"_"+datatype.lower()+".csv"))
        print("feature dir", feature_dir)
        #print("labels", ld_labels)
        ld_labels = pd.read_csv(ld_labels)
        for path, folders, files in os.walk(feature_dir):
            for f in folders:
                f_path=str(os.path.join(path, f, "features.npy"))
                if not os.path.isfile(f_path):
                    print("ISFILE check in work")
                    continue

                # Get labels if possible  
                iD = "-".join(str(f).split("-")[:3])             
                if ld_name=="TCGA":
                    pid = "bcr_patient_barcode"
                    use_label = "OS.time"
                elif ld_name in ["S36", "UNN"]:
                    iD = int(iD.split("_")[0].split("-")[1])
                    pid = "patient_id"
                    use_label = "OS.time"
                elif ld_name == "NLST":
                    iD = int(iD.split("_")[0])
                    pid = "pid"
                    use_label = "OS.time"
                if ld_name=="UNN":
                    use_label = "days_to_death"
                l = ld_labels[ld_labels[pid]==iD]
                #print(f"ID {iD}, has labels w shape {l.shape}")
                if l.shape[0]==0:
                    #No label for this slide, so we skip it.
                    continue
                l_val = l[use_label].values.item()
                if np.isnan(l_val):
                    continue

                # Load features
                np_features = np.load(f_path, mmap_mode='r')
                n_features = use_features # How many tiles from this slide (typically max 1000)
                if use_features > np_features.shape[0]:
                    n_features = np_features.shape[0]
                    print("Dont have this many tiles. Use", np_features.shape[0])

                indices = np.arange(len(np_features))
                np.random.shuffle(indices)
                indices = indices[:n_features]
                np_features = np_features[indices]
                #print("features dim", np_features.shape)

                # rand_index = random.sample(range(np_features.shape[0]), n_features)
                # np_features = np_features[rand_index]
                domains[ld_name] += np_features.shape[0]

                # Save OS.time labels
                labels += [int(l_val)]*np_features.shape[0]
                # Subtype labels
                is_lusc = 1 if "LUSC" in datatype else 0
                _indexes += np_features.shape[0] * [is_lusc]
                # Stage labels
                k = use_label_dict[ld_name]
                stage_val = l[k].values.item()
                if ld_name=="NLST":
                    stage_val = nlst_stage[stage_val]
                stage_val = stage_val.split(" ")[-1].upper()
                stage_val = stage_val.translate(remove_digits)
                stage_labels += [stage_val]*np_features.shape[0]

                # Remove 3 cols w metadata
                np_features = np_features[:, 3:]

                # Add corresponding slide
                slides_id.append([str(os.path.join(path, f))]*n_features)

                # If we didnt use n_features==use_features
                if n_features < use_features:
                    # pad the indices with -1
                    diff = use_features-n_features
                    addon = np.zeros((diff,)) - 1
                    indices = np.hstack((indices,addon))

                # tile_counter += n_features
                # if tile_counter > 3232:
                #     print("slide counter", tile_counter)

                # Save selected features and indices of tiles from the slides
                if features.shape[0]==0:
                    # 1st iteration
                    features = np_features
                    slides_id_tileindexes = indices
                else:
                    features = np.concatenate((features, np_features), axis=0)
                    slides_id_tileindexes = np.vstack((slides_id_tileindexes, indices)) # i,use_features
        print(features.shape)
    
    # Flatten slides id
    slides_id = list(np.asarray(slides_id).flatten())
    return features, labels, _indexes, domains, stage_labels, slides_id, slides_id_tileindexes


def feature_to_slide(feat):
    """
    feat: a string containing TCGA/(R46/UNN)/(S36/Mainz). or a list
    Returns a dictionary id_to_place with key=slide id, value=folder to find that slide
    """
    id_to_place = {}
    if type(feat)==str:
        if "TCGA" in feat:
            base_dir = "/home/vilde/data/TCGA/TCGA-LUSC/Diagnostic_slides"
        elif "UNN" in feat or "R46" in feat:
            base_dir = "/home/vilde/data/UNN"
        elif "Mainz" in feat or "S36" in feat:
            base_dir = "/home/vilde/data/S36"

        for root,dirs,files in os.walk(base_dir):
            for name in files:
                if name[-4:]==".svs":
                    id_to_place[name] = root
    elif type(feat)==list:
        if any("TCGA" in f for f in feat):
            base_dir = "/home/vilde/data/TCGA/TCGA-LUSC/Diagnostic_slides"
            for root,dirs,files in os.walk(base_dir):
                for name in files:
                    if name[-4:]==".svs":
                        id_to_place[name] = root
        if any("UNN" in f for f in feat) or any("R46" in f for f in feat):
            base_dir = "/home/vilde/data/UNN"
            for root,dirs,files in os.walk(base_dir):
                for name in files:
                    if name[-4:]==".svs":
                        id_to_place[name] = root
        if any("Mainz" in f for f in feat) or any("S36" in f for f in feat):
            base_dir = "/home/vilde/data/S36"
            for root,dirs,files in os.walk(base_dir):
                for name in files:
                    if name[-4:]==".svs":
                        id_to_place[name] = root
    return id_to_place


def plot_tiles(features, slides_id, slides_id_tileindexes, indices, distances=[], identification="", extractor="", cluster_id=0, tuned_dir="", repeat=0, save_path="figures/slide_clusters"):
    """
    Use the indexes to plot the tiles corresponding to these features

    features                : array tiles,feature_dim
    slides_id               : list: tiles
    slides_id_tileindexes   : array slides,n_tiles. Index of -1 means padding (no tile available). Connection between features order and np.load(feature_dir) order
    indices                 : list: max length tiles. Tile indexes (which tiles are in the cluster eg). Based on arrangement in features
    """
    #print("in plot tiles")
    if np.where(slides_id_tileindexes==-1)[0].size==0:  # or useslides_id_tileindexes.size == features.shape[0]:
        # Easy version - each slide has exactly n_iles tiles registered
        n_tiles = slides_id_tileindexes.shape[-1]
        use_slides = [slides_id[i] for i in indices] # list of actual slides
        slides_places = feature_to_slide(use_slides)

        use_slide_ids = [s.split("/")[-1] for s in use_slides]
        use_slide_ids = ["-".join(a.replace("_", "-").split("-")[:3]) for a in use_slide_ids]
        #print(f"Cluster {cluster_id} uses slides {use_slide_ids}")
        saving_names = []
        tiles = []
        for i in range(len(indices)):
            index = indices[i]
            all_tile_index = slides_id_tileindexes.flatten()[index]
            tile_feature = np.load(slides_id[index]+"/features.npy")[all_tile_index,:]
            tile_metadata = tile_feature[:3] # tile_level, x_coordinate, y_coordinate
            tile_level, x, y = tile_metadata
            tile_level,x,y = int(tile_level), int(x), int(y)

            use_slide = slides_id[index].split("/")[-1] # Just the ID, not full path to feature file
            slide = openslide.open_slide(Path(slides_places[use_slide]) / use_slide)
            dzg = DeepZoomGenerator(slide, tile_size=224, overlap=0)
            tile = dzg.get_tile(level=tile_level, address=(x,y))
            r = "repeat"+str(repeat)
            save_at = str(Path(save_path) / extractor / tuned_dir / r / str(cluster_id) / identification)
            
            Path(save_at).mkdir(parents=True, exist_ok=True)

            save_name = use_slide_ids[i]+f"_d{distances[i]:.2f}"+".png"
            if save_name in saving_names:
                save_name = use_slide_ids[i]+f"_t2_d{distances[i]:.2f}"+".png"
            saving_names.append(save_name)
            tile.save(save_at+"/"+ save_name)
            tiles.append(tile)

        rows=2
        cols = (len(tiles)+rows-1) // rows
        d_to_d = {"TCGA": "TCGA", "S36": "Mainz", "R46": "UNN"}
        #fig, ax = plt.subplots(rows,cols,figsize=(10,8))
        if len(tiles) < 5:
            fig, ax = plt.subplots(rows,cols,figsize=(7,8))
        if len(tiles) > 10:
            fig, ax = plt.subplots(rows,cols,figsize=(16,8))
        else:
            fig, ax = plt.subplots(rows,cols,figsize=(14,8))
        ax = ax.flatten()
        plt.rc('font', family='serif',size=24)
        for i,img in enumerate(tiles):
            ax[i].imshow(np.asarray(img))
            ax[i].axis('off')
            dd = saving_names[i].split("_")[0].split("-")[0]
            ax[i].set_title(d_to_d[dd])
        for i in range(len(tiles),len(ax)):
            fig.delaxes(ax[i])
        plt.tight_layout()
        save_at = str(Path(save_path) / extractor / tuned_dir / r)     
        plt.savefig(save_at+str(cluster_id)+"_full.png")

        for i in range(len(tiles)):
            tiles[i].close()

        plt.close()

    else:
        print("plot tiles is not implemented for large n_features yet (where not all slides have that many tiles)")


def do_kmeans(features, slides_id, slides_id_tileindexes, n_clusters=1000, identification="", extractor="", tuned_dir="", repeat=0):
    """
    features                : array all_tiles,feature_dim
    slides_id               : list: all_tiles
    slides_id_tileindexes   : array slides,n_tiles. Index of -1 means padding (no tile available)
    """
    #print("Perform kmeans on features of shape", features.shape)
    kmean = KMeans(n_clusters=n_clusters, init='k-means++', n_init='auto')
    #features = preprocessing.normalize(features)
    kmean.fit(features) #preprocessing.normalize(
    cluster_centers = kmean.cluster_centers_ # shape n_clusters,feature_dim
    labels = kmean.labels_ # shape n_features,
    
    _,counts = np.unique(labels, return_counts=True)
    common = np.argsort(counts)[::-1]
    # Cluster number common[0] is the most common. 
    # It has counts[common[0]] entries.
    s = [counts[c] for c in common]
    if repeat <3:
        print("Cluster sizes by size:", s[:50])
        print(f"Smallest cluster has {np.min(s)} entries")
    d = kmean.transform(features) # Shape all_tiles,n_clusters. Distances to cluster centers

    def vis_closest_diffdata(cluster_nr, d, cluster_id, labels, tiles_per_dataset=1, saveat=""):
        """
        d: kmean.transform(features)
        Plot the closest tiles to the cluster center - from different datasets
        But make sure they are in the cluster
        """
        c = cluster_nr
        distances = d[:,c] # Distances to cluster center c. Shape all_tiles,
        ind = np.argsort(distances)
        ind = ind[labels[ind]==c] # Only consider tiles that are classified to be in this cluster
        used = {"TCGA":0, "UNN":0, "S36":0}
        use_indices = []
        distance_to_center = []

        for i in ind:
            # Which dataset does this index (tile) correspond to?
            used_slide = slides_id[i].split("/")[-1]
            if "TCGA" in used_slide and used["TCGA"]<tiles_per_dataset:
                # This is the TCGA slide closest to the centroid
                use_indices.append(i)
                distance_to_center.append(distances[i])
                used["TCGA"] += 1
                #print("tile", i)
            elif ("UNN" in used_slide) and used["UNN"]<tiles_per_dataset:
                use_indices.append(i)
                distance_to_center.append(distances[i])
                used["UNN"] += 1
                #print("tile", i)
            elif ("S36" in used_slide) and used["S36"]<tiles_per_dataset:
                use_indices.append(i)
                distance_to_center.append(distances[i])
                used["S36"] += 1
                #print("tile", i)
            elif not (("S36" in used_slide) or ("UNN" in used_slide) or ("TCGA" in used_slide)):
                # Could not recognize slide as either of the 3 datasets
                print("Slide used", used_slide, "does not belong to TCGA, UNN or S36?")

            if len(use_indices)>=3*tiles_per_dataset:
                plot_tiles(features, slides_id, slides_id_tileindexes, indices=use_indices, distances = distance_to_center, identification=identification, extractor=extractor, cluster_id=cluster_id, tuned_dir=tuned_dir, repeat=repeat, save_path=saveat)
                return
        # There are not enough tiles from all datasets in the cluster, so we just plot what we have
        plot_tiles(features, slides_id, slides_id_tileindexes, indices=use_indices, distances = distance_to_center, identification=identification, extractor=extractor, cluster_id=cluster_id, tuned_dir=tuned_dir, repeat=repeat, save_path=saveat)

    def vis_closest(cluster_nr, d, cluster_id, tiles=5, saveat=""):
        """
        d: kmean.transform(features)
        """
        c = cluster_nr
        distances = d[:,c] # Distances to cluster center c. Shape all_tiles,
        ind = np.argsort(distances)
        use_indices = []
        distance_to_center = []
        for i in ind:
            # Which dataset does this index (tile) correspond to?
            used_slide = slides_id[i].split("/")[-1]
            use_indices.append(i)
            distance_to_center.append(distances[i])
        
            if len(use_indices)>=tiles:
                plot_tiles(features, slides_id, slides_id_tileindexes, indices=use_indices, distances=distance_to_center, identification=identification, extractor=extractor, cluster_id=cluster_id, tuned_dir=tuned_dir, repeat=repeat, save_path=saveat)
                return

    def vis_outliers(cluster_nr, d, cluster_id, labels, tiles=5, saveat="", midliers=False):
        c = cluster_nr
        distances = d[:,c] # Distances to cluster center c. Shape all_tiles,
        ind = np.argsort(distances)[::-1] # Descending order (longest distance first)
        ind = ind[labels[ind]==c] #Keep only indices that are in the actual cluster
        if midliers:
            cutoff = int(len(ind)/2) - 2
            ind = ind[cutoff:]
            saveat = saveat.replace("outliers", "midliers")
        use_indices = []
        distance_to_center = []
        for i in ind:
            # Which dataset does this index (tile) correspond to?
            used_slide = slides_id[i].split("/")[-1]
            use_indices.append(i)
            distance_to_center.append(distances[i])
        
            if len(use_indices)>=tiles:
                plot_tiles(features, slides_id, slides_id_tileindexes, indices=use_indices, distances=distance_to_center, identification=identification, extractor=extractor, cluster_id=cluster_id, tuned_dir=tuned_dir, repeat=repeat, save_path=saveat)
                return

    # For the 5 most common clusters, show 2 closest entries from each dataset
    if repeat < 4:
        for i in range(5):
            # Plot closest from different datasets. Then from any dataset
            # Common clusters
            j = i # for slide clusters com
            cluster_nr = common[j] # The i'th most common cluster
            vis_closest_diffdata(cluster_nr, d, cluster_id=j, labels=labels, tiles_per_dataset=4, saveat="figures/slide_clusters_v2/slide_clusterscom/closest_dataspread")
            vis_closest(cluster_nr, d, cluster_id=j, tiles=10, saveat="figures/slide_clusters_v2/slide_clusterscom/closest_general")
            vis_outliers(cluster_nr, d, cluster_id=j, labels=labels, tiles=10, saveat="figures/slide_clusters_v2/slide_clusterscom/outliers")
            vis_outliers(cluster_nr, d, cluster_id=j, labels=labels, tiles=10, saveat="figures/slide_clusters_v2/slide_clusterscom/outliers", midliers=True)

            # Mid clusters
            j = int(i+len(common)//2) # for slide_clusters mid
            cluster_nr = common[j] # The i'th most common cluster
            vis_closest_diffdata(cluster_nr, d, cluster_id=j, labels=labels, tiles_per_dataset=4, saveat="figures/slide_clusters_v2/slide_clustersmid/closest_dataspread")
            vis_closest(cluster_nr, d, cluster_id=j, tiles=10, saveat="figures/slide_clusters_v2/slide_clustersmid/closest_general")
            vis_outliers(cluster_nr, d, cluster_id=j, labels=labels, tiles=10, saveat="figures/slide_clusters_v2/slide_clustersmid/outliers")
            vis_outliers(cluster_nr, d, cluster_id=j, labels=labels, tiles=10, saveat="figures/slide_clusters_v2/slide_clustersmid/outliers", midliers=True)

            # Rare clusters
            j = -i-1 # slide_clusters rare
            cluster_nr = common[j] # The i'th most common cluster
            vis_closest_diffdata(cluster_nr, d, cluster_id=j, labels=labels, tiles_per_dataset=4, saveat="figures/slide_clusters_v2/slide_clustersrare/closest_dataspread")
            vis_closest(cluster_nr, d, cluster_id=j, tiles=10, saveat="figures/slide_clusters_v2/slide_clustersrare/closest_general")
            vis_outliers(cluster_nr, d, cluster_id=j, labels=labels, tiles=10, saveat="figures/slide_clusters_v2/slide_clustersrare/outliers")
            vis_outliers(cluster_nr, d, cluster_id=j, labels=labels, tiles=10, saveat="figures/slide_clusters_v2/slide_clustersrare/outliers", midliers=True)

    purity_scores = []
    for j in range(len(common)):
        cluster_nr = common[j]
        c = np.where(labels==cluster_nr)[0].astype(int) # Indexes belonging to this cluster_nr
        c_dataset = [slides_id[i].split("/")[-1].split('-')[0] for i in c]
        elements, local_counts = np.unique(c_dataset, return_counts=True)

        # Purity score is frequency of the most dominant class
        dominant_dataset = elements[np.argmax(local_counts)]
        purity_score = np.max(local_counts) / len(c_dataset)
        purity_scores.append(purity_score)
        #print("Purity score:", purity_score, ": ", dominant_dataset)
    print("Purity scores: avg, full:", np.mean(purity_scores))

    # Calculate cluster metrics
    db = davies_bouldin_score(features, labels)
    ch = calinski_harabasz_score(features, labels)
    sil = silhouette_score(features, labels)
    print(f"Other cluster metrics:, {db:.3f}, {ch:.0f}, {sil:.4f}")

    # Save the cluster in a npz file
    save_cluster = str(Path("clusters") / extractor / tuned_dir)
    Path(save_cluster).mkdir(parents=True, exist_ok=True)
    save_cluster += f"/{repeat}.npz"
    cluster = {}
    cluster["cluster_centers"] = cluster_centers
    cluster["labels"] = labels
    cluster["distances"] = d # all_tiles,n_clusters: Distance from each tile to each cluster center.
    cluster["purity_scores"] = np.asarray(purity_scores)
    np.savez(save_cluster, **cluster)

    if (repeat == 0) and False:
        # Visualize everything
        print("pca visualization")
        #kmean = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4)
        reduced_data = PCA(n_components=2).fit_transform(features)
        kmean = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        kmean.fit(reduced_data)
        h = 0.01
        xmin, xmax = reduced_data[:,0].min()-0.1, reduced_data[:,0].max()+0.1
        ymin, ymax = reduced_data[:,1].min()-0.1, reduced_data[:,1].max()+0.1
        xx, yy = np.meshgrid(np.arange(xmin,xmax,h), np.arange(ymin,ymax,h))
        kmean.cluster_centers_ = kmean.cluster_centers_.astype(float)
        Z = kmean.predict(np.c_[xx.ravel(), yy.ravel()])
        #Color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z,interpolation="nearest",extent=(xx.min(),xx.max(),yy.min(),yy.max()), cmap=plt.cm.Paired, aspect="auto", origin="lower")
        plt.plot(reduced_data[:,0], reduced_data[:,1], "ko", markersize=1)
        #white centroids
        centroids = kmean.cluster_centers_
        plt.scatter(centroids[:,0], centroids[:,1], marker="x", s=169,linewidths=3,color="w",zorder=10)
        plt.title("kmeans clustering")
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        plt.xticks()
        plt.yticks()
        plt.savefig(f"figures/slide_clusters_v2/slide_clusters/kmeans_visualization_{extractor}_{identification}.png")

    return np.mean(purity_scores), sil, ch, db

if __name__ == "__main__":
    model_type="v2_tuned" #{"iBOTViTBasePANCAN", "uni", "virchow", "virchow2", "phikon2", "phikon_tuned", "v2_tuned", "p2_tuned"}
    extractor = "Virchow2 tuneS f100c200"
    datatype="lusc" # {luad, lusc, all}
    #tuned_dir = "phikon_tuned/dino_p_tcga_ha4"
    #tuned_dir = "phikon_tuned/dino_p_s36_ha"
    tuned_dir = "v2_tuned/dino_v2_s36_ha"
    #tuned_dir = "p2_tuned/dino_p2_s36_ha"
    use_features=100 #10 -> 10200 total tiles
    n_clusters=200 # 200
    identification=""
    n_repeats = 20
    #save_dir = ""

    if not "tuned" in model_type:
        tuned_dir = ""

    features, labels, subtype_indexes, domains, stage_labels, slides_id, slides_id_tileindexes = extract_features(model=model_type, datatype=datatype, tuned_dir=tuned_dir, use_features=use_features)
    # features: n,feature_dim
    tuned_dir = tuned_dir.split("/")[-1]
    purities = np.zeros((n_repeats))
    sils = []
    chs = []
    dbs = []
    for repeat in range(n_repeats):
        purity_scores, sil, ch, db = do_kmeans(features, slides_id, slides_id_tileindexes, n_clusters=n_clusters, extractor=extractor, identification=identification, tuned_dir=tuned_dir, repeat=repeat)
        purities[repeat] = purity_scores
        sils.append(sil)
        chs.append(ch)
        dbs.append(db)

    print(f"Final average purity score for {extractor}, {tuned_dir}: {np.mean(purities):.3f}: over {repeat+1} repetitions")
    print(f"{np.mean(purities)} +- {np.std(purities)}")
    print("Avg Silhouette score:", np.mean(sils))
    print("Avg CH score:", np.mean(chs))
    print("Avg DB score:", np.mean(dbs))