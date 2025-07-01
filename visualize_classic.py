import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import openslide
import os
import pandas as pd
import random
import sys

from pathlib import Path

from rl_benchmarks.constants import PREPROCESSED_DATA_DIR

def extract_features(model="iBOTViTBasePANCAN", datatype="all", tuned_dir="phikon_tuned/dinov2_tcga_phikon"):
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
        print("labels", ld_labels)
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
                #n_features = np_features.shape[0]
                n_features = 10

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

                # Save selected features
                if features.shape[0]==0:
                    # 1st iteration
                    features = np_features
                else:
                    features = np.concatenate((features, np_features), axis=0)
        print(features.shape)
    return features, labels, _indexes, domains, stage_labels

def features_red(features, type_red="TSNE", n_components=2):
    """
    INPUTS:
        type_red: str. Which type of feature reduction to perform. (TSNE; PCA; UMAP)
        n_components: int. How many dimensions for output embeddings
    """
    print("Features going into reduction are shaped ", features.shape)

    if "tsne" in type_red.lower():
        from sklearn.manifold import TSNE
        print("start tsne")
        X_embedded = TSNE(n_components=n_components, learning_rate='auto', init='random', perplexity=30).fit_transform(features)
        print("tsne transformed!")
        return X_embedded, None

    elif "pca" in type_red.lower():
        from sklearn.decomposition import PCA
        print("start pca")
        X_trans = PCA(n_components=n_components)
        X_trans.fit(features)
        X_embedded = X_trans.transform(features)
        print("pca transformed!")
        return X_embedded, X_trans

    elif "umap" in type_red.lower():
        import umap
        print("start umap")
        # May adjust n_neighbors param
        X_trans = umap.UMAP(n_components=n_components)
        X_trans.fit(features)
        X_embedded = X_trans.transform(features)
        return X_embedded, X_trans

    else:
        print(f"Feature reduction {type_red} is not implemented in features_red()")
        sys.exit()

def plot_red(X_embedded, domains, labels, subtype_indexes, type_red="TSNE", plot_components=[0,1], model="iBOTViTBasePANCAN", col="cite", datatype="lusc", tuned_dir="phikon_tuned/dinov2_tcga_phikon", stage_labels=None):
    """
    INPUTS:
        X_embedded
        domains: dict. Info of how many datapoints from each dataset.
        labels: 
        type_red: str. Which type of reduction has been applied
        plot_components: list[int]. List of dimensions of the reduction to plot. len(plot_components) <= n_components of feature reduction alg.
        model: str. Which model has created the features
        col: what labels to use for plot. {cite, dss, subtype}
    """
    model_to_type = {"iBOTViTBasePANCAN":"Phikon", "uni":"UNI", "virchow":"Virchow", "virchow2":"Virchow 2", "phikon2":"Phikon 2", "phikon_tuned": "Phikon tuned", "v2_tuned": "Virchow2 tuned", "p2_tuned": "Phikon2 tuned"}
    feature_type = model_to_type[model]
    savename = "".join([d[0] for d in feature_type.lower().split(" ")])
    if "tuned" in model:
        savename = tuned_dir.split("/")[-1] + "_tuned"
    print("savename", savename)

    fig, ax = plt.subplots()
    fig.set_size_inches((4, 3))

    plot_dim = len(plot_components)
    sc_list = []

    if plot_dim == 2:
        col1 = plot_components[0]
        col2 = plot_components[1]

        if col=="cite":
            alpha = 0.7
            s=2
            for key, value in domains.items():
                if key=="TCGA":
                    m="*"
                    a=0
                    b=domains["TCGA"]
                    color='y'
                elif key=="S36":
                    m="."
                    a=domains["TCGA"]
                    b=a+domains["S36"]
                    color='g'
                elif key=="UNN":
                    m="x"
                    a=domains["TCGA"]+domains["S36"]
                    b=a+domains["UNN"]
                    color='b'
                elif key=="NLST":
                    m="v"
                    a=domains["TCGA"]+domains["S36"]+domains["UNN"]
                    b=a+domains["NLST"]
                    color='c'
                sc = ax.scatter(X_embedded[a:b, col1], X_embedded[a:b, col2], c=[color]*len(X_embedded[a:b, col2]), s=s, alpha=alpha)#*col, marker=m)
                sc_list.append(sc)

        elif col=="dss":
            for key, value in domains.items():
                color='y'
                if key=="TCGA":
                    m="*"
                    a=0
                    b=domains["TCGA"]
                elif key=="S36":
                    m="."
                    a=domains["TCGA"]
                    b=a+domains["S36"]
                elif key=="UNN":
                    m="x"
                    a=domains["TCGA"]+domains["S36"]
                    b=a+domains["UNN"]
                elif key=="NLST":
                    m="v"
                    a=domains["TCGA"]+domains["S36"]+domains["UNN"]
                    b=a+domains["NLST"]
                    color='c'
                sc = ax.scatter(X_embedded[a:b, col1], X_embedded[a:b, col2], c=labels[a:b], s=2, marker=m, alpha=0.7)#*col, marker=m)
                sc_list.append(sc)

        elif col=="subtype":
            for key, value in domains.items():
                if key=="TCGA":
                    m="*"
                    a=0
                    b=domains["TCGA"]
                elif key=="S36":
                    m="."
                    a=domains["TCGA"]
                    b=a+domains["S36"]
                elif key=="UNN":
                    m="x"
                    a=domains["TCGA"]+domains["S36"]
                    b=a+domains["UNN"]
                elif key=="NLST":
                    m="v"
                    a=domains["TCGA"]+domains["S36"]+domains["UNN"]
                    b=a+domains["NLST"]
                colors = ['purple', 'pink']
                sc = ax.scatter(X_embedded[a:b, col1], X_embedded[a:b, col2], c=[colors[i] for i in subtype_indexes[a:b]], s=2, marker=m, alpha=0.7)#*col, marker=m)
                sc_list.append(sc)

        elif col=="stage":
            alpha=0.03
            size=70
            # Plot background colors of dataset first
            for key, value in domains.items():
                if key=="TCGA":
                    a=0
                    b=domains["TCGA"]
                    color='teal'
                elif key=="S36":
                    a=domains["TCGA"]
                    b=a+domains["S36"]
                    color='mediumvioletred'
                elif key=="UNN":
                    a=domains["TCGA"]+domains["S36"]
                    b=a+domains["UNN"]
                    color='darkorange'
                elif key=="NLST":
                    a=domains["TCGA"]+domains["S36"]+domains["UNN"]
                    b=a+domains["NLST"]
                    color='c'
                sc = ax.scatter(X_embedded[a:b, col1], X_embedded[a:b, col2], c=[color]*len(X_embedded[a:b, col2]), s=size, alpha=alpha, edgecolors='none')#*col, marker=m)
                sc_list.append(sc)

            #Plot stage colors
            convert_labels = {'III':'III', 'IIIA':'III', 'IIB':'IIB', 'IIIC':'III', 'IA':'IA', 'II':'II', 'IVA':'IV', 'I':'I', 'IB':'IB', '[DISCREPANCY]':'other', 'IIIB':'III', 'IV':'IV', 'IIA':'IIA', 'IVA':'IV', 'IVB':'IV', 'IIIC':'III'}
            convert_labels = {'III':'III', 'IIIA':'III', 'IIB':'II', 'IIIC':'III', 'IA':'I', 'II':'II', 'IVA':'IV', 'I':'I', 'IB':'I', '[DISCREPANCY]':'other', 'IIIB':'III', 'IV':'IV', 'IIA':'II', 'IVA':'IV', 'IVB':'IV', 'IIIC':'III'}
            stage_labels = [convert_labels[l] for l in stage_labels]
            print("new stage labels set:", set(stage_labels))

            colors_dict = {'I':'g', 'IA': 'limegreen', 'IB': 'y', 'II':'c', 'IIA': 'deepskyblue', 'IIB': 'b', 'III':'purple', 'IV':'r', 'other': 'gray'}
            colors_dict = {'I':'g', 'II':'limegreen', 'III':'y', 'IV':'yellow', 'other': 'gray'}
            colors = ['g','limegreen','y','c','deepskyblue','b','purple','r','gray']
            colors = ['g','c','purple','r','gray']
            colors = ['g', 'limegreen', 'y', 'yellow', 'gray']
            cmap = matplotlib.colors.ListedColormap(colors)
            #['I', 'IA', 'IB', 'II', 'IIA', 'IIB', 'III', 'IV', 'other']

            # print("coolwarm cmap has N =", plt.cm.coolwarm.N) # 256
            # cmap = plt.cm.coolwarm #jet
            # cmaplist = [cmap(i) for i in range(cmap.N)]
            # cmap = cmap.from_list('custom cmap', cmaplist, cmap.N)
            # print("New cmap N = ", cmap.N)
            N = len(set(convert_labels.values()))
            bounds = np.linspace(0,N,N+1)
            norm = matplotlib.colors.BoundaryNorm(bounds, N)
            # label_to_int = dict(zip(list(set(convert_labels.values())), np.arange(N)))
            # stage_int_labels = [label_to_int[s] for s in stage_labels]

            for key, value in domains.items():
                color='y'
                if key=="TCGA":
                    m="*"
                    a=0
                    b=domains["TCGA"]
                    s=6
                elif key=="S36":
                    a=domains["TCGA"]
                    b=a+domains["S36"]
                elif key=="UNN":
                    a=domains["TCGA"]+domains["S36"]
                    b=a+domains["UNN"]
                elif key=="NLST":
                    a=domains["TCGA"]+domains["S36"]+domains["UNN"]
                    b=a+domains["NLST"]
                
                cc = [colors_dict[s] for s in stage_labels[a:b]]

                sc = ax.scatter(X_embedded[a:b, col1], X_embedded[a:b, col2], c=cc, cmap=cmap, norm=norm, s=1)
                sc_list.append(sc)

        else:
            print(f"Plot type {col} is not available in plot_red")

        plt.grid(False)
        ax.set_xlim((np.min(X_embedded[:,col1])*1.1, np.max(X_embedded[:,col1])*1.1))
        ax.set_ylim((np.min(X_embedded[:,col2])*1.1, np.max(X_embedded[:,col2])*1.1))
        ax.set_xticks([])
        ax.set_yticks([])

    elif plot_dim == 3:
        savename += "_dim3"
        #Create 3d plot
        ax = fig.add_subplot(111, projection='3d')
        col1 = plot_components[0]
        col2 = plot_components[1]
        col3 = plot_components[2]
        
        if col=="cite":
            for key, value in domains.items():
                if key=="TCGA":
                    m="*"
                    a=0
                    b=domains["TCGA"]
                    color='y'
                elif key=="S36":
                    m="."
                    a=domains["TCGA"]
                    b=a+domains["S36"]
                    color='g'
                elif key=="UNN":
                    m="x"
                    a=domains["TCGA"]+domains["S36"]
                    b=a+domains["UNN"]
                    color='b'
                elif key=="NLST":
                    m="v"
                    a=domains["TCGA"]+domains["S36"]+domains["UNN"]
                    b=a+domains["NLST"]
                    color='c'
                sc = ax.scatter(X_embedded[a:b, col1], X_embedded[a:b, col2], X_embedded[a:b, col3], c=[color]*len(X_embedded[a:b, col2]), s=2, alpha=0.7)
                sc_list.append(sc)

        elif col=="dss":
            for key, value in domains.items():
                color='y'
                if key=="TCGA":
                    m="*"
                    a=0
                    b=domains["TCGA"]
                elif key=="S36":
                    m="."
                    a=domains["TCGA"]
                    b=a+domains["S36"]
                elif key=="UNN":
                    m="x"
                    a=domains["TCGA"]+domains["S36"]
                    b=a+domains["UNN"]
                elif key=="NLST":
                    m="v"
                    a=domains["TCGA"]+domains["S36"]+domains["UNN"]
                    b=a+domains["NLST"]
                    color='c'
                sc = ax.scatter(X_embedded[a:b, col1], X_embedded[a:b, col2], X_embedded[a:b, col3], c=labels[a:b], s=2, marker=m, alpha=0.7)#*col, marker=m)
                sc_list.append(sc)

        elif col=="subtype":
            for key, value in domains.items():
                if key=="TCGA":
                    m="*"
                    a=0
                    b=domains["TCGA"]
                elif key=="S36":
                    m="."
                    a=domains["TCGA"]
                    b=a+domains["S36"]
                elif key=="UNN":
                    m="x"
                    a=domains["TCGA"]+domains["S36"]
                    b=a+domains["UNN"]
                elif key=="NLST":
                    m="v"
                    a=domains["TCGA"]+domains["S36"]+domains["UNN"]
                    b=a+domains["NLST"]
                colors = ['purple', 'pink']
                sc = ax.scatter(X_embedded[a:b, col1], X_embedded[a:b, col2], X_embedded[a:b, col3], c=[colors[i] for i in subtype_indexes[a:b]], s=2, marker=m, alpha=0.7)#*col, marker=m)
                sc_list.append(sc)

        else:
            print(f"Plot type {col} is not available in plot_red")

        plt.grid(False)
        ax.set_xlim((np.min(X_embedded[:,col1])*1.1, np.max(X_embedded[:,col1])*1.1))
        ax.set_ylim((np.min(X_embedded[:,col2])*1.1, np.max(X_embedded[:,col2])*1.1))
        ax.set_zlim((np.min(X_embedded[:,col3])*1.1, np.max(X_embedded[:,col3])*1.1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    if datatype=="all":
        datatype="luadlusc"

    if type_red=="TSNE":
        type_red = "t-SNE"

    if col=="cite":
        legend = ax.legend(handles=sc_list, labels=["TCGA", "Mainz", "UNN"], loc="lower left", title="Dataset", framealpha=0.5)
        #legend.get_frame().set_facecolor('none')
        legend.get_frame().set_linewidth(0.0)
        ax.add_artist(legend)

        if plot_dim == 3:
            i=0
            for angle in range(0,360*3+1, 45):
                angle_n = (angle+180) % 360 - 180
                elev = azim = roll = 0
                if angle <= 360:
                    elev = angle_n
                elif angle <= 360*2:
                    azim = angle_n
                # elif angle < 360*3:
                #     roll = angle_n
                else:
                    elev = azim = roll = angle_n
                ax.view_init(elev=elev, azim=azim)
                plt.tight_layout()
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                plt.savefig(f"figures/features/dim3/{type_red}_"+savename+f"_cite_{datatype}_{i}.png", dpi=300)
                i+=1

        plt.title(f"{type_red} of "+feature_type+" features")
        plt.tight_layout()
        plt.savefig(f"figures/features/{type_red}_"+savename+f"_cite_{datatype}.png", dpi=500)
        fig.clear()
        print(f"{type_red} cite plot done")

    elif col=="dss":
        legend2 = ax.legend(sc_list, domains, loc="lower right", title="Dataset")
        ax.add_artist(legend2)
        legend2.get_frame().set_facecolor('none')
        legend2.get_frame().set_linewidth(0.0)
        fig.colorbar(sc_list[0])

        plt.title(f"{type_red} of "+feature_type+" features")
        plt.tight_layout()
        plt.savefig(f"figures/features/{type_red}_"+savename+f"_dss_{datatype}.png", dpi=500)
        fig.clear()
        print(f"{type_red} dss plot done")

    elif col=="subtype":
        legend2 = ax.legend(sc_list, domains, loc="lower right", title="Dataset")
        ax.add_artist(legend2)
        legend2.get_frame().set_facecolor('none')
        legend2.get_frame().set_linewidth(0.0)

        plt.title(f"{type_red} of "+feature_type+" features")
        plt.tight_layout()
        plt.savefig(f"figures/features/{type_red}_"+savename+f"_subtypes.png", dpi=500)
        fig.clear()
        print("subtype plot done")

    elif col=="stage":
        l1 = matplotlib.patches.Patch(color='teal', label="TCGA")
        l2 = matplotlib.patches.Patch(color='mediumvioletred', label='Mainz')
        l3 = matplotlib.patches.Patch(color='darkorange', label='UNN')
        legend = ax.legend(handles=[l1, l2, l3], loc=0, title="Dataset", framealpha=0.7)

        #legend = ax.legend(handles=sc_list, labels=["TCGA", "Mainz", "UNN"], loc=0, title="Dataset", framealpha=0.7)
        legend.get_frame().set_linewidth(0.0)
        ax.add_artist(legend)

        stages = ['I', 'IA', 'IB', 'II', 'IIA', 'IIB', 'III', 'IV', 'other']
        stages = ['I', 'II', 'III', 'IV'] #, 'other']
        N = len(stages)
    
        norm = matplotlib.colors.BoundaryNorm(np.linspace(0,N,N+1), N) # stages+'other'
        cmap = matplotlib.colors.ListedColormap(['g','limegreen','y','yellow','gray'])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = plt.colorbar(sm, spacing='uniform', label="Pathologic stage")
        cb.set_ticks((np.arange(N)+0.5))
        cb.set_ticklabels(stages)

        # import IPython
        # IPython.embed()

        plt.title(f"{type_red} of "+feature_type+" features")
        plt.tight_layout()
        plt.savefig(f"figures/features/{type_red}_"+savename+f"_stage_{datatype}.png", dpi=500)
        fig.clear()
        print(f"{type_red} stage plot done")

    return


def plot_random_slides(dataset="TCGA"):
    # Random slides
    #dataset = "TCGA/TCGA-LUSC/Diagnostic_slides"
    #dataset="NLST"
    slides_dir = "/home/vilde/data/"+dataset
    if dataset=="S36":
        dataset = "Mainz"
    if "Diagnostic_slides" in dataset:
        dataset = dataset.split("/")[1]

    slides = []
    for path, folders, files in os.walk(slides_dir):
        slides.extend(os.path.join(path, file) for file in files if file.endswith(".svs")) 
    print(f"Found {len(slides)} slides")

    num_examples = 3
    plot_slides = random.choices(slides, k=num_examples)
    slide_imgs = []

    plt.figure(figsize=(6,9))
    for i, slide in enumerate(plot_slides):
        s = openslide.open_slide(slide)
        try:
            s_img = s.associated_images['thumbnail']
        except:
            print("Using macro image!")
            s_img = s.associated_images['macro']
        #s_img.save("figures/features/img"+str(i)+".png")
        plt.subplot(num_examples,1,i+1)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        plt.imshow(s_img)
        plt.xticks([])

    plt.suptitle(f"{dataset} slide examples", size='xx-large')
    plt.tight_layout()
    plt.savefig(f"figures/features/{dataset.lower()}_examples.png", dpi=800)


if __name__ == "__main__":
    model_type="phikon_tuned" #{"iBOTViTBasePANCAN", "uni", "virchow", "virchow2", "phikon2", "phikon_tuned", "v2_tuned", "p2_tuned"}
    datatype="lusc" # {luad, lusc, all}
    type_reds=["UMAP", "TSNE", "PCA"] # "TSNE", "UMAP", "PCA"
    plot_cols=["cite"] # {cite, dss, subtype, stage}
    n_components=2
    plot_components=[0,1]
    # tuned_dir = "phikon_tuned/dino_p_tcga_ha4"
    tuned_dir = "phikon_tuned/dino_p_s36_ha"
    # tuned_dir = "v2_tuned/dino_v2_tcga_ha"
    # tuned_dir = "p2_tuned/dino_p2_unn_ha"
    # dino_p_tcga_ha4  dino2_p_tcga_ha
    # dino_p_unn_ha

    addon=""
    if "tuned" in model_type:
        addon = "_"+tuned_dir.split("_")[-2]

    features, labels, subtype_indexes, domains, stage_labels = extract_features(model=model_type, datatype=datatype, tuned_dir=tuned_dir)
    print("Stage labels", list(set(stage_labels)))
    for type_red in type_reds:
        X_emb, X_trans = features_red(features=features, type_red=type_red, n_components=n_components)
        # Save X_emb to csv file
        dataset_list = ["0"]*domains["TCGA"] + ["1"]*domains["S36"] + ["2"]*domains["UNN"]
        save_dict = {'x': X_emb[:,0], 'y': X_emb[:,1], 'dataset':dataset_list}
        save_dict = pd.DataFrame.from_dict(save_dict)
        save_dict.to_csv(f"figures/features/csv_files/{model_type}{addon}_{type_red}.csv", index=False)
        print(f"csv file saved to {model_type}{addon}_{type_red}.csv")
        for cp in plot_cols:
            plot_red(X_embedded=X_emb, domains=domains, labels=labels, subtype_indexes=subtype_indexes, type_red=type_red, plot_components=plot_components, model=model_type, col=cp, datatype=datatype, tuned_dir=tuned_dir, stage_labels=stage_labels)