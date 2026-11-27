# Third-party code, models and data

flybrain's own code (install.sh, physics/, web/, tools/, docker files, docs) is released under the MIT
License (see LICENSE). `brain/` is released under GPL-2.0-or-later (see brain/LICENSE), because it derives from GPL code.

## Why brain/ is GPL

- `brain/app/model.py` says it is a direct transcription of the update step in `reference/run_pytorch.py`.
  That file comes from eonsystemspbc/fly-brain and is GPL-2.0-or-later (see `reference/LICENSE`). Because the brain code
  derives from it, `brain/` is distributed under GPL-2.0-or-later, not MIT. It runs as a separate service over HTTP/WebSocket.

## Data (downloaded by install.sh, not redistributed)

install.sh downloads these files from the public sources below at the pinned commits and checks each one against
`data/SHA256SUMS`. The data is not part of this repository.

| File in data/ | Source (pinned) | Terms as published | sha256 |
|---|---|---|---|
| `2025_Connectivity_783.parquet` (100.8 MB) | github.com/philshiu/Drosophila_brain_model, `Connectivity_783.parquet` @ 91bdd1e | The repository's LICENSE is MIT and covers its code. The README states no separate license for the data. FlyWire's guidelines state that its public release data is CC BY-NC 4.0. A Zenodo record of the v783 release (10.5281/zenodo.10676866, different file format) states CC BY 4.0. | efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347 |
| `2025_Completeness_783.csv` | same repository, `Completeness_783.csv` @ 91bdd1e | same as above. Line endings converted from LF to CRLF so the bytes match the copy that was tested. Same rows and order. | 52b0ac6094cd32c546f8d4c341e094376f48f4e791f8db9b166de5dff8199ea4 |
| `sez_neurons.pickle` | same repository @ 91bdd1e | same as above. Not read by the running stack. | 2bc5a2a8289924f2de39a8e68bdc07025cb6d6c6c6ddc83a973c0cace7cec2ce |
| `positions/Supplemental_file1_neuron_annotations.tsv` | github.com/flyconnectome/flywire_annotations, `supplemental_files/Supplemental_file1_neuron_annotations.tsv` @ a83b277 (v3.2.0) | The repository has no LICENSE file (GitHub reports none). Its README asks users to cite the authors listed for v3.2.0 and later (see below). Used by the dev tools only. | b214970b55d2fbe0853bba536fdcb9e28730f4eb7ab06f600491df795da683cd |

Citations for the data:

- Shiu et al. (2024). A Drosophila computational brain model reveals sensorimotor processing. *Nature* 634, 210-219. doi:10.1038/s41586-024-07763-9
- Dorkenwald et al. (2024). Neuronal wiring diagram of an adult brain. *Nature* 634, 124-138. doi:10.1038/s41586-024-07558-y
- Annotation table: the flywire_annotations README asks for Berg et al. (2026), Schlegel et al. (2024), Tastekin et al. (2026), Matsliah et al. (2024) and Dorkenwald et al. (2024) for v3.2.0 and later. Full references are in that README.

## Code and models from others

- **reference/run_pytorch.py, reference/UPSTREAM_README.md, reference/LICENSE**: eonsystemspbc/fly-brain, GPL-2.0 (GitHub license
  metadata). See the open question above.
- **reference/rojas/**: MIT, Copyright (c) 2026 Enrique Manuel Rojas Aliaga (license text in reference/rojas/LICENSE).
  Ported from rjo6615/fly-brain-interactive (MIT).
- **physics/app/data/groom_clip.npz**: grooming kinematics derived from the NeuroMechFly repository (NeLy-EPFL), Apache-2.0.
  Copy of the license in physics/app/data/NeuroMechFly-LICENSE-Apache-2.0.txt, notes in physics/app/data/NOTICE.md.
  Cite Lobato-Rios et al. (2022). NeuroMechFly, a neuromechanical model of adult Drosophila melanogaster. *Nature Methods* 19, 620-627. doi:10.1038/s41592-022-01466-7
- **flygym 2.1.1** (NeLy-EPFL/flygym, Apache-2.0) and **MuJoCo** (Apache-2.0): the physics runtime. The fly body meshes come from the flygym package.
- **three.js** (MIT), **SvelteKit** (MIT), **Vite** (MIT): the web UI.
- **FastAPI** (MIT), **uvicorn** (BSD-3-Clause), **PyTorch** (BSD-style), **NumPy** (BSD-style), **Triton** (MIT),
  **Apache Arrow / pyarrow** (Apache-2.0): Python runtime. Each keeps its own license; see the upstream projects.

## Written in this repository

- The procedural kitchen, the fly-world logic and the UI are generated in code. No third-party 3D assets are shipped.
- Parts of the code were written with AI assistance.
