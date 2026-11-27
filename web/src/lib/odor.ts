import * as THREE from 'three';
import { fillHeat, validGrid, type OdorGrid, type OdorSpecies } from './odorheat';

/** Translucent heat texture lying on the counter plane (z = 0). Grid is row-major c[j*nx+i], i = x, j = y (physics odor.snapshot). */
export class OdorOverlay {
  readonly mesh: THREE.Mesh;
  private tex?: THREE.DataTexture;
  private dims = '';
  constructor(scene: THREE.Scene) {
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({
      transparent: true, depthWrite: false, toneMapped: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2
    }));
    this.mesh.renderOrder = 2;
    this.mesh.position.z = 0.4;   // just above the counter top
    this.mesh.visible = false;
    scene.add(this.mesh);
  }
  /** Returns true if something changed (caller re-renders). */
  set(g: OdorGrid | null, species: OdorSpecies): boolean {
    if (!g || !validGrid(g)) { const was = this.mesh.visible; this.mesh.visible = false; return was; }
    const dims = g.nx + 'x' + g.ny;
    if (dims !== this.dims || !this.tex) {
      this.tex?.dispose();
      const t = new THREE.DataTexture(new Uint8Array(g.nx * g.ny * 4), g.nx, g.ny, THREE.RGBAFormat);
      t.magFilter = THREE.LinearFilter; t.minFilter = THREE.LinearFilter; t.colorSpace = THREE.SRGBColorSpace;
      t.generateMipmaps = false; t.needsUpdate = true;
      this.tex = t; this.dims = dims;
      (this.mesh.material as THREE.MeshBasicMaterial).map = t;
      (this.mesh.material as THREE.MeshBasicMaterial).needsUpdate = true;
    }
    fillHeat(g, species, this.tex.image.data as Uint8Array);
    this.tex.needsUpdate = true;
    // Cell i is centred at x0 + i*cell (up to a sub-cell offset of cell/8), so the plane spans the grid centres plus half a cell each side.
    const w = g.nx * g.cell_mm, h = g.ny * g.cell_mm;
    this.mesh.scale.set(w, h, 1);
    this.mesh.position.x = g.x0_mm + (g.nx - 1) * g.cell_mm / 2;
    this.mesh.position.y = g.y0_mm + (g.ny - 1) * g.cell_mm / 2;
    this.mesh.visible = true;
    return true;
  }
  dispose() { this.tex?.dispose(); (this.mesh.material as THREE.Material).dispose(); this.mesh.geometry.dispose(); this.mesh.removeFromParent(); }
}
