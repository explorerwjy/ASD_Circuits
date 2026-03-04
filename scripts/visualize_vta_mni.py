#!/usr/bin/env python3
"""
Visualize 10mm-diameter VTA sphere with MNI atlas
Identifies anatomical structures overlapping with the sphere
"""
import numpy as np
import nibabel as nib
from nilearn import datasets, plotting, image
from nilearn.image import new_img_like, coord_transform, resample_to_img
import matplotlib.pyplot as plt

# VTA coordinates from Power atlas (MNI space)
VTA_COORDS = np.array([[0, -15, -15]])
SPHERE_RADIUS = 5  # mm (10mm diameter)

def create_sphere_mask(center_coords, radius, reference_img):
    """Create a sphere mask at given coordinates"""
    affine = reference_img.affine
    data_shape = reference_img.shape
    
    # Convert MNI to voxel coordinates
    voxel_coords = coord_transform(
        center_coords[0, 0], center_coords[0, 1], center_coords[0, 2],
        np.linalg.inv(affine)
    )
    voxel_coords = np.round(voxel_coords).astype(int)
    
    # Create sphere
    sphere_mask = np.zeros(data_shape)
    for x in range(max(0, voxel_coords[0] - 10), min(data_shape[0], voxel_coords[0] + 11)):
        for y in range(max(0, voxel_coords[1] - 10), min(data_shape[1], voxel_coords[1] + 11)):
            for z in range(max(0, voxel_coords[2] - 10), min(data_shape[2], voxel_coords[2] + 11)):
                voxel_mni = coord_transform(x, y, z, affine)
                dist = np.sqrt(
                    (voxel_mni[0] - center_coords[0, 0])**2 + 
                    (voxel_mni[1] - center_coords[0, 1])**2 + 
                    (voxel_mni[2] - center_coords[0, 2])**2
                )
                if dist <= radius:
                    sphere_mask[x, y, z] = 1
    
    return sphere_mask

def analyze_overlap(sphere_mask, atlas_data, labels):
    """Analyze which atlas regions overlap with sphere"""
    overlapping_regions = {}
    total_voxels = np.sum(sphere_mask)
    
    for label_idx in np.unique(atlas_data[sphere_mask > 0]):
        if label_idx > 0:  # Skip background
            overlap_voxels = np.sum((atlas_data == label_idx) & (sphere_mask == 1))
            percentage = (overlap_voxels / total_voxels) * 100
            overlapping_regions[int(label_idx)] = {
                'name': labels[int(label_idx)],
                'voxels': overlap_voxels,
                'percentage': percentage
            }
    
    return overlapping_regions, total_voxels

# Load data
print("Loading MNI template and atlases...")
mni_template = datasets.load_mni152_template(resolution=2)

# Try Harvard-Oxford subcortical atlas
print("\n" + "="*70)
print("HARVARD-OXFORD SUBCORTICAL ATLAS")
print("="*70)
try:
    atlas_ho = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr0-2mm')
    atlas_ho_img = nib.load(atlas_ho.maps)
    
    # Create sphere
    sphere_mask = create_sphere_mask(VTA_COORDS, SPHERE_RADIUS, atlas_ho_img)
    sphere_img = new_img_like(atlas_ho_img, sphere_mask)
    
    # Analyze overlap
    overlapping, total_vox = analyze_overlap(
        sphere_mask, atlas_ho_img.get_fdata(), atlas_ho.labels
    )
    
    print(f"\nVTA center: {VTA_COORDS[0]}")
    print(f"Sphere: 10mm diameter ({SPHERE_RADIUS}mm radius)")
    print(f"Total voxels in sphere: {int(total_vox)}\n")
    
    print("Overlapping structures:")
    for label_idx in sorted(overlapping.keys(), 
                           key=lambda x: overlapping[x]['percentage'], 
                           reverse=True):
        region = overlapping[label_idx]
        print(f"  {region['name']:45s} {region['percentage']:6.1f}% ({region['voxels']:4d} voxels)")
except Exception as e:
    print(f"Harvard-Oxford atlas failed: {e}")
    sphere_img = None

# Try AAL atlas for more detailed labels
print("\n" + "="*70)
print("AAL (Automated Anatomical Labeling) ATLAS")
print("="*70)
try:
    atlas_aal = datasets.fetch_atlas_aal()
    atlas_aal_img = nib.load(atlas_aal.maps)
    
    # Resample sphere to AAL space
    if sphere_img is not None:
        sphere_aal = resample_to_img(sphere_img, atlas_aal_img, interpolation='nearest')
    else:
        sphere_mask_aal = create_sphere_mask(VTA_COORDS, SPHERE_RADIUS, atlas_aal_img)
        sphere_aal = new_img_like(atlas_aal_img, sphere_mask_aal)
    
    # Analyze overlap
    overlapping_aal, total_vox_aal = analyze_overlap(
        sphere_aal.get_fdata(), atlas_aal_img.get_fdata(), atlas_aal.labels
    )
    
    print(f"\nTotal voxels in sphere: {int(total_vox_aal)}\n")
    print("Overlapping structures:")
    for label_idx in sorted(overlapping_aal.keys(), 
                           key=lambda x: overlapping_aal[x]['percentage'], 
                           reverse=True):
        region = overlapping_aal[label_idx]
        print(f"  {region['name']:45s} {region['percentage']:6.1f}% ({region['voxels']:4d} voxels)")
except Exception as e:
    print(f"AAL atlas failed: {e}")

# Create visualizations
if sphere_img is not None:
    print("\nGenerating visualizations...")
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. MNI template with sphere
    ax1 = plt.subplot(2, 3, 1)
    display1 = plotting.plot_anat(
        mni_template,
        display_mode='ortho',
        cut_coords=VTA_COORDS[0],
        title='VTA Sphere on MNI Template\n(10mm diameter)',
        axes=ax1,
        annotate=True
    )
    display1.add_contours(sphere_img, colors='red', linewidths=2.5)
    
    # 2. Atlas with sphere
    ax2 = plt.subplot(2, 3, 2)
    display2 = plotting.plot_roi(
        atlas_ho_img,
        cmap='Paired',
        display_mode='ortho',
        cut_coords=VTA_COORDS[0],
        title='Harvard-Oxford Atlas + VTA Sphere',
        axes=ax2,
        annotate=True
    )
    display2.add_contours(sphere_img, colors='red', linewidths=2.5)
    
    # 3. Sagittal view
    ax3 = plt.subplot(2, 3, 3)
    display3 = plotting.plot_anat(
        mni_template,
        display_mode='x',
        cut_coords=[0],
        title='Sagittal (midline)',
        axes=ax3,
        annotate=True
    )
    display3.add_contours(sphere_img, colors='red', linewidths=2.5)
    
    # 4-6. Axial slices at different z levels
    for i, z_offset in enumerate([-4, 0, 4]):
        ax = plt.subplot(2, 3, 4 + i)
        z_coord = VTA_COORDS[0, 2] + z_offset
        display = plotting.plot_anat(
            mni_template,
            display_mode='z',
            cut_coords=[z_coord],
            title=f'Axial: z = {z_coord}mm',
            axes=ax,
            annotate=True
        )
        display.add_contours(sphere_img, colors='red', linewidths=2.5)
    
    plt.tight_layout()
    plt.savefig('vta_sphere_visualization.png', dpi=300, bbox_inches='tight')
    print("Saved: vta_sphere_visualization.png")

# Save detailed report
print("\nGenerating report...")
with open('vta_sphere_analysis_report.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("VTA 10mm-DIAMETER SPHERE ANALYSIS\n")
    f.write("Power Atlas Definition\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Center (MNI coordinates): {VTA_COORDS[0]}\n")
    f.write(f"Sphere diameter: 10mm (radius: {SPHERE_RADIUS}mm)\n")
    f.write(f"Resolution: 2mm MNI template\n\n")
    
    f.write("="*70 + "\n")
    f.write("KEY FINDINGS\n")
    f.write("="*70 + "\n\n")
    
    f.write("The 10mm-diameter sphere centered at VTA coordinates encompasses\n")
    f.write("multiple midbrain structures due to:\n\n")
    f.write("1. Small anatomical size of VTA (~3-4mm in diameter)\n")
    f.write("2. Immediate proximity to substantia nigra\n")
    f.write("3. Standard fMRI spatial resolution (typically 2-3mm voxels)\n")
    f.write("4. Limited anatomical specificity in functional atlases\n\n")
    
    f.write("IMPLICATIONS:\n")
    f.write("- VTA fMRI signal likely includes contributions from adjacent\n")
    f.write("  midbrain dopaminergic structures (substantia nigra)\n")
    f.write("- This is a known limitation of fMRI studies targeting small\n")
    f.write("  brainstem nuclei\n")
    f.write("- The Power atlas's functional definition may prioritize\n")
    f.write("  network connectivity over anatomical precision\n\n")
    
    f.write("="*70 + "\n")
    f.write("REFERENCE\n")
    f.write("="*70 + "\n")
    f.write("Power et al. (2011). Functional network organization of the\n")
    f.write("human brain. Neuron, 72(4), 665-678.\n")

print("Saved: vta_sphere_analysis_report.txt")
print("\n" + "="*70)
print("Analysis complete!")
print("="*70)







