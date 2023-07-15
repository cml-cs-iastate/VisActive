

def coverage_and_uniqueness(output, num_classes, num_vc, vc_per_class, top_k):
    """
    Calculate support and uniqueness
    """

    # initialize the coverage and uniqueness
    coverage = dict()                                                  # {VC: value}}
    uniqueness = dict()                                                # {VC: value}}
    closest_images_per_vc = {i: 0 for i in range(num_vc)}              # {VC: count}
    docs_per_class = {c: 0 for c in range(num_classes)}                # {class: count}
    vc_in_class = {i: None for i in range(num_vc)}                    # {class: {VC: {class1, class2}}}

    # ------------------------------------------------------------------------------------------------------------------

    for img_name, info in output.items():
        # Run for the correctly predicted images only
        # Only calculate this for VisActive, not VisActive
        # if info['class'] != info['prediction']:
        #     continue

        # Calculate the number of documents per class
        docs_per_class[info['class']] += 1

        # Calculate the closest image per each VC
        # Only with in the VC that belongs to that class
        c = info['class']
        range_ = [j for j in range(c * vc_per_class, (c + 1) * vc_per_class)]
        for i, vc in enumerate(info['sorted_vc'][:vc_per_class]):
            if vc in range_ and i < top_k:
                closest_images_per_vc[vc] += 1

        # Calculate the number of classes VC is in a top k
        for vc in info['sorted_vc'][:1]:
            if vc_in_class[vc] is None:
                vc_in_class[vc] = set()
            if c not in vc_in_class[vc]:
                vc_in_class[vc].add(c)

    for k, v in docs_per_class.items():
        print(f'class {k} has {v} images')

    # Calculate support and uniqueness
    for i in range(num_vc):
        c = int(i // vc_per_class)
        coverage[i] = closest_images_per_vc[i] / docs_per_class[c] if docs_per_class[c] > 0 else 0
        uniqueness[i] = (1 / num_classes) + (1 - (len(vc_in_class[i]) / num_classes)) if vc_in_class[i] is not None else 0

        print(f'Class {c} ...\t VC{i} ...\t Coverage {round(coverage[i], 3)}\t ... Uniqueness {round(uniqueness[i], 3)}')

    return coverage, uniqueness
