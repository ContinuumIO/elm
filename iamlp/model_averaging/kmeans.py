import numpy as np
def kmeans_model_averaging(models, no_shuffle=1):
    print('kmeans_model_averaging')
    models.sort(key=lambda x:x.inertia_)
    centroids = [model.cluster_centers_ for model in models]
    models_improved = models[:no_shuffle]
    probs = np.linspace(len(models) + 1, 1, len(models))
    probs = probs / np.sum(probs)
    centroid_idxes = list(range(len(models)))
    for idx in range(no_shuffle, len(models)):
        model = models[idx]
        has_seen = set()
        cent = 0
        while cent < model.cluster_centers_.shape[0]:
            model_choice = int(np.random.choice(centroid_idxes, p=probs))
            centroid_choice = int(np.random.randint(0, centroids[model_choice].shape[1], 1))
            choice = (model_choice, centroid_choice)
            if models[centroid_choice].class_pcents_[centroid_choice] < require_pcent:
                has_seen.add(choice)
                continue
            if not choice in has_seen:
                has_seen.add(choice)
                model.cluster_centers_[cent, :] = centroids[model_choice][centroid_choice, :]
                cent += 1
        delattr(model, 'inertia_')
        models_improved.append(model)
    return models_improved
