# Adapted from: https://thepythoncode.com/code/generate-visualize-text-embeddings-python

import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console
# from rich.table import Table
from rich.panel import Panel
import argparse
import struct

parser = argparse.ArgumentParser(description='Generate and analyze sentence embeddings')
parser.add_argument('--model_name', type=str, default="LinYuting/sentence_transformer_128_dim_model", help='Name of the sentence transformer model')
parser.add_argument('--output_file', type=str, default="embeddings.fvecs", help='File to save output embeddings')
args = parser.parse_args()

console = Console()

# ═══════════════════════════════════════════════════════════════
# 1. DIVERSE SENTENCE CORPUS — 50 sentences, 10 categories
# ═══════════════════════════════════════════════════════════════
console.print(Panel("[bold cyan]Step 1: Building Sentence Corpus[/bold cyan]", border_style="blue"))

sentences = [
    # Technology (5)
    "The computer processed data at incredible speed",
    "Machine learning models require large amounts of training data",
    "Python is widely used for artificial intelligence applications",
    "Cloud computing enables scalable web services",
    "The algorithm optimized the search results efficiently",
    # Nature & Animals (5)
    "The dog chased the ball across the green field",
    "Cats are independent creatures that enjoy solitude",
    "The majestic eagle soared high above the mountains",
    "Dolphins are highly intelligent marine mammals",
    "The tiger stalked its prey through the dense jungle",
    # Food & Cooking (5)
    "The chef prepared a delicious Italian pasta dish",
    "Fresh ingredients make the best homemade meals",
    "The chocolate cake was rich and decadently sweet",
    "Grilling steak requires high heat and proper timing",
    "Japanese sushi demands precise knife skills and fresh fish",
    # Travel & Places (5)
    "The ancient ruins attracted tourists from around the world",
    "Paris is known as the city of love and romance",
    "The tropical beach had crystal clear turquoise water",
    "Mountain climbers reached the summit after days of effort",
    "The bustling city never sleeps with its vibrant nightlife",
    # Emotions & Relationships (5)
    "She felt overwhelming joy when she received the good news",
    "Heartbreak can feel like a physical pain in your chest",
    "Their friendship had lasted through decades of ups and downs",
    "Pride swelled in his chest as he watched his daughter graduate",
    "Anxiety crept in as the deadline approached rapidly",
    # Science & Knowledge (5)
    "The scientist conducted experiments to test the hypothesis",
    "Mathematics is the language of the universe",
    "Quantum physics challenges our understanding of reality",
    "DNA contains the genetic blueprint of all living organisms",
    "The theory of evolution explains the diversity of life",
    # Sports & Activities (5)
    "The soccer team celebrated their championship victory",
    "Swimming is an excellent full-body cardiovascular workout",
    "The marathon runner crossed the finish line exhausted but proud",
    "Basketball requires both athleticism and strategic thinking",
    "Yoga combines physical poses with breathing and meditation",
    # Art & Creativity (5)
    "The painter captured the sunset in brilliant orange and red hues",
    "Music has the power to evoke deep emotional responses",
    "The novelist spent years crafting the perfect ending",
    "Dance allows expression beyond what words can convey",
    "Photography freezes a single moment for eternity",
    # Business & Work (5)
    "The startup raised millions in venture capital funding",
    "Effective leadership requires both vision and empathy",
    "The company announced record profits for the fiscal year",
    "Remote work has transformed the modern workplace",
    "Negotiation skills are essential for closing major deals",
    # Health & Medicine (5)
    "Regular exercise reduces the risk of heart disease",
    "The doctor prescribed antibiotics for the bacterial infection",
    "Mental health is just as important as physical health",
    "Vaccines have saved millions of lives throughout history",
    "A balanced diet provides essential nutrients for the body",
]

categories = (
    ["Tech"] * 5 + ["Animals"] * 5 + ["Food"] * 5 + ["Travel"] * 5 +
    ["Emotions"] * 5 + ["Science"] * 5 + ["Sports"] * 5 + ["Art"] * 5 +
    ["Business"] * 5 + ["Health"] * 5
)

console.print(f"[green]Loaded {len(sentences)} sentences across {len(set(categories))} categories[/green]")

# ═══════════════════════════════════════════════════════════════
# 2. GENERATE EMBEDDINGS
# ═══════════════════════════════════════════════════════════════
console.print(f"\n[bold cyan]Step 2: Generating Embeddings[/bold cyan]")

# model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer(args.model_name)
embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)

console.print(f"[green]Model: {args.model_name}[/green]")
console.print(f"  Shape: [yellow]{embeddings.shape}[/yellow]")
console.print(f"  First embedding (8 of {embeddings.shape[1]} dims): [yellow]{embeddings[0][:8].round(4)}[/yellow]")

# Save embeddings as fvecs file
# with open(f"{args.output_file}", "wb") as f:
#     f.write(np.array([embeddings.shape[1]], dtype=np.int32).tobytes())
#     f.write(embeddings.astype(np.float32).tobytes())

with open(f"{args.output_file}", 'wb') as f:
    for vec in embeddings:
        dim = vec.shape[0]
        f.write(struct.pack('<i', dim)) # Write dimension as int32
        f.write(vec.astype('<f4').tobytes()) # Write vector values as float32

# # ═══════════════════════════════════════════════════════════════
# # 3. PCA VISUALIZATION
# # ═══════════════════════════════════════════════════════════════
# console.print(f"\n[bold cyan]Step 3: PCA Visualization[/bold cyan]")

# pca = PCA(n_components=2, random_state=42)
# embeddings_2d_pca = pca.fit_transform(embeddings)

# cat_colors = {
#     "Tech": "#3b82f6", "Animals": "#10b981", "Food": "#f59e0b",
#     "Travel": "#8b5cf6", "Emotions": "#ef4444", "Science": "#06b6d4",
#     "Sports": "#f97316", "Art": "#ec4899", "Business": "#6366f1", "Health": "#14b8a6",
# }

# fig, ax = plt.subplots(figsize=(16, 11))
# for cat in sorted(set(categories)):
#     mask = [c == cat for c in categories]
#     ax.scatter(embeddings_2d_pca[mask, 0], embeddings_2d_pca[mask, 1],
#                c=cat_colors[cat], label=cat, alpha=0.75, s=120,
#                edgecolors='white', linewidth=1.2, zorder=2)
#     for i, is_cat in enumerate(mask):
#         if is_cat:
#             ax.annotate(sentences[i][:40] + "...",
#                        (embeddings_2d_pca[i, 0], embeddings_2d_pca[i, 1]),
#                        fontsize=6, alpha=0.6, ha='center', va='bottom',
#                        xytext=(0, -8), textcoords='offset points')

# ax.set_title(f"Text Embeddings Visualized with PCA\n"
#              f"50 sentences -> 384-dim vectors -> 2D projection\n"
#              f"PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
#              f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%",
#              fontsize=13, fontweight='bold', pad=15)
# ax.legend(loc='upper left', framealpha=0.9, fontsize=9, ncol=2)
# ax.grid(True, alpha=0.15)
# plt.tight_layout()
# plt.savefig('01_pca_visualization.png', dpi=180, bbox_inches='tight')
# plt.close()
# console.print("[green]PCA plot saved[/green]")

# # ═══════════════════════════════════════════════════════════════
# # 4. T-SNE VISUALIZATION
# # ═══════════════════════════════════════════════════════════════
# console.print(f"\n[bold cyan]Step 4: t-SNE Visualization[/bold cyan]")

# tsne = TSNE(n_components=2, perplexity=8, random_state=42, max_iter=1000)
# embeddings_2d_tsne = tsne.fit_transform(embeddings)

# fig, ax = plt.subplots(figsize=(16, 11))
# for cat in sorted(set(categories)):
#     mask = [c == cat for c in categories]
#     ax.scatter(embeddings_2d_tsne[mask, 0], embeddings_2d_tsne[mask, 1],
#                c=cat_colors[cat], label=cat, alpha=0.75, s=120,
#                edgecolors='white', linewidth=1.2, zorder=2)
#     for i, is_cat in enumerate(mask):
#         if is_cat:
#             ax.annotate(sentences[i][:35] + "...",
#                        (embeddings_2d_tsne[i, 0], embeddings_2d_tsne[i, 1]),
#                        fontsize=6, alpha=0.55, ha='center', va='bottom',
#                        xytext=(0, -8), textcoords='offset points')

# ax.set_title("Text Embeddings Visualized with t-SNE\n"
#              "t-SNE preserves local structure — similar sentences cluster tightly",
#              fontsize=13, fontweight='bold', pad=15)
# ax.legend(loc='upper left', framealpha=0.9, fontsize=9, ncol=2)
# ax.grid(True, alpha=0.15)
# plt.tight_layout()
# plt.savefig('02_tsne_visualization.png', dpi=180, bbox_inches='tight')
# plt.close()
# console.print("[green]t-SNE plot saved[/green]")

# # ═══════════════════════════════════════════════════════════════
# # 5. COSINE SIMILARITY HEATMAP
# # ═══════════════════════════════════════════════════════════════
# console.print(f"\n[bold cyan]Step 5: Cosine Similarity Heatmap[/bold cyan]")

# indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
# subset_sentences = [sentences[i] for i in indices]
# subset_embeddings = embeddings[indices]
# subset_cats = [categories[i] for i in indices]
# sim_matrix = cosine_similarity(subset_embeddings)

# fig, ax = plt.subplots(figsize=(14, 12))
# sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
#             xticklabels=[f"[{c}] {s[:25]}..." for c, s in zip(subset_cats, subset_sentences)],
#             yticklabels=[f"[{c}] {s[:25]}..." for c, s in zip(subset_cats, subset_sentences)],
#             vmin=0, vmax=1, linewidths=0.5, linecolor='white',
#             cbar_kws={'label': 'Cosine Similarity', 'shrink': 0.8}, ax=ax)
# ax.set_title("Cosine Similarity Between Sentence Embeddings\n"
#              "1.0 = identical meaning, 0.0 = completely unrelated",
#              fontsize=13, fontweight='bold', pad=15)
# plt.xticks(rotation=45, ha='right', fontsize=8)
# plt.yticks(rotation=0, fontsize=8)
# plt.tight_layout()
# plt.savefig('03_similarity_heatmap.png', dpi=180, bbox_inches='tight')
# plt.close()
# console.print("[green]Similarity heatmap saved[/green]")

# # ═══════════════════════════════════════════════════════════════
# # 6. DIMENSION ANALYSIS
# # ═══════════════════════════════════════════════════════════════
# console.print(f"\n[bold cyan]Step 6: Dimension Analysis[/bold cyan]")

# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# # Value distribution
# ax = axes[0]
# ax.hist(embeddings.flatten(), bins=80, color='#3b82f6', alpha=0.7, edgecolor='white', linewidth=0.3)
# ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
# ax.set_title("Distribution of All Embedding Values\n(50 sentences x 384 dims = 19,200 values)",
#              fontsize=11, fontweight='bold')
# ax.set_xlabel("Embedding Value"); ax.set_ylabel("Frequency")
# ax.grid(True, alpha=0.2)

# # Per-dimension statistics
# ax = axes[1]
# dim_means = embeddings.mean(axis=0)
# dim_stds = embeddings.std(axis=0)
# dims = np.arange(len(dim_means))
# ax.fill_between(dims, dim_means - dim_stds, dim_means + dim_stds, alpha=0.3, color='#3b82f6')
# ax.plot(dims, dim_means, color='#1d4ed8', linewidth=0.8)
# ax.axhline(y=0, color='red', linestyle='--', alpha=0.4)
# ax.set_title("Per-Dimension Statistics\n(Mean across 50 sentences)", fontsize=11, fontweight='bold')
# ax.set_xlabel("Dimension Index (0-383)"); ax.set_ylabel("Value")
# ax.grid(True, alpha=0.2)

# # Cumulative explained variance
# ax = axes[2]
# pca_full = PCA().fit(embeddings)
# cumsum = np.cumsum(pca_full.explained_variance_ratio_)
# d50 = np.argmax(cumsum >= 0.50) + 1
# d90 = np.argmax(cumsum >= 0.90) + 1
# ax.plot(cumsum, color='#3b82f6', linewidth=2)
# ax.axhline(y=0.50, color='#f59e0b', linestyle='--', alpha=0.6)
# ax.axhline(y=0.90, color='#ef4444', linestyle='--', alpha=0.6)
# ax.set_title(f"Cumulative Explained Variance\n{d50} dims -> 50%, {d90} dims -> 90%",
#              fontsize=11, fontweight='bold')
# ax.set_xlabel("Number of Principal Components"); ax.set_ylabel("Cumulative Variance")
# ax.grid(True, alpha=0.2); ax.set_xlim(0, 120)

# plt.tight_layout()
# plt.savefig('04_dimension_analysis.png', dpi=180, bbox_inches='tight')
# plt.close()
# console.print(f"[green]Dimension analysis saved[/green]")
# console.print(f"  [yellow]{d50}[/yellow] dimensions capture 50% of variance")
# console.print(f"  [yellow]{d90}[/yellow] dimensions capture 90% of variance")

# # ═══════════════════════════════════════════════════════════════
# # 7. SEMANTIC SIMILARITY DEMO
# # ═══════════════════════════════════════════════════════════════
# console.print(f"\n[bold cyan]Step 7: Semantic Similarity Demo[/bold cyan]")

# demo_pairs = [
#     ("The dog played in the park", "A canine ran through the green field"),
#     ("The dog played in the park", "The stock market crashed yesterday"),
#     ("I love eating pizza and pasta", "Italian cuisine is my favorite food"),
#     ("I love eating pizza and pasta", "The spaceship launched into orbit"),
#     ("She felt incredibly happy today", "Joy radiated from her entire being"),
#     ("She felt incredibly happy today", "The computer needs a software update"),
#     ("Machine learning is transforming industries", "AI and deep learning are reshaping business"),
#     ("Machine learning is transforming industries", "The cat napped in the warm sunlight"),
# ]

# table = Table(show_header=True, header_style="bold white")
# table.add_column("Sentence A", style="cyan", max_width=35)
# table.add_column("Sentence B", style="white", max_width=35)
# table.add_column("Sim", style="yellow", width=8)
# table.add_column("Relation", width=10)

# for a, b in demo_pairs:
#     ea = model.encode([a], normalize_embeddings=True)[0]
#     eb = model.encode([b], normalize_embeddings=True)[0]
#     sim = float(np.dot(ea, eb))
#     rel = "[green]SAME[/green]" if sim > 0.5 else "[red]DIFF[/red]"
#     color = "green" if sim > 0.5 else "red"
#     table.add_row(a[:35], b[:35], f"[{color}]{sim*100:.1f}%[/{color}]", rel)

# console.print(table)

# # Find most similar/different pairs
# full_sim = cosine_similarity(embeddings)
# pairs = [(full_sim[i][j], i, j) for i in range(len(sentences)) for j in range(i+1, len(sentences))]
# pairs.sort(reverse=True)

# console.print("\n[bold green]Most Similar Pairs:[/bold green]")
# for sim, i, j in pairs[:3]:
#     console.print(f'  [green]{sim*100:.1f}%[/green] — "{sentences[i][:50]}" <-> "{sentences[j][:50]}"')

# console.print("\n[bold red]Most Different Pairs:[/bold red]")
# for sim, i, j in pairs[-3:]:
#     console.print(f'  [red]{sim*100:.1f}%[/red] — "{sentences[i][:50]}" <-> "{sentences[j][:50]}"')

# console.print(f"\n[bold green]{'='*60}[/bold green]")
# console.print("[bold green]Embedding Analysis — Complete![/bold green]")
# console.print(f"[dim]Plots saved: 01_pca, 02_tsne, 03_heatmap, 04_dimensions[/dim]")
