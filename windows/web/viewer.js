import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

const viewer = document.querySelector('#viewer');
let scene;
let camera;
let renderer;
let controls;

function setup() {
  if (renderer) {
    if (!renderer.domElement.isConnected) viewer.replaceChildren(renderer.domElement);
    return;
  }
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x090d13);
  camera = new THREE.PerspectiveCamera(55, viewer.clientWidth / viewer.clientHeight, 0.01, 1000);
  camera.position.set(1, 1, 1);
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(viewer.clientWidth, viewer.clientHeight);
  viewer.append(renderer.domElement);
  controls = new OrbitControls(camera, renderer.domElement);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 2));
  function render() {
    requestAnimationFrame(render);
    controls.update();
    renderer.render(scene, camera);
  }
  render();
}

async function artifact(id, name, headers) {
  const response = await fetch(`/api/v2/sessions/${id}/artifacts/${name}`, { headers: headers() });
  return response.ok ? response : null;
}

export async function viewSession(id, { headers, job, artifactNames = ['pointcloud.ply', 'mesh_mesh_clean.ply'] }) {
  setup();
  while (scene.children.length > 1) scene.remove(scene.children.at(-1));
  let response = null;
  let selectedName = '';
  for (const name of artifactNames) {
    response = await artifact(id, name, headers);
    if (response) {
      selectedName = name;
      break;
    }
  }
  if (!response) {
    job.textContent = 'No point cloud or mesh artifact is available yet';
    return;
  }
  const geometry = new PLYLoader().parse(await response.arrayBuffer());
  geometry.computeBoundingSphere();
  const mesh = selectedName.includes('mesh_');
  const material = mesh
    ? new THREE.MeshStandardMaterial({ vertexColors: geometry.hasAttribute('color'), color: 0x78aaff, side: THREE.DoubleSide })
    : geometry.hasAttribute('color')
      ? new THREE.PointsMaterial({ size: 0.004, vertexColors: true })
      : new THREE.PointsMaterial({ size: 0.004, color: 0x66ccff });
  scene.add(mesh ? new THREE.Mesh(geometry, material) : new THREE.Points(geometry, material));
  controls.target.copy(geometry.boundingSphere.center);
  camera.position.copy(geometry.boundingSphere.center).addScalar(Math.max(geometry.boundingSphere.radius * 2, 0.5));
  job.textContent = `Viewing ${id}`;
}

export async function showMask(id, name, { headers, job }) {
  const response = await artifact(id, `object/masks/${name}`, headers);
  if (!response) {
    job.textContent = 'Foreground mask is unavailable';
    return;
  }
  const image = document.createElement('img');
  image.alt = `Foreground mask ${name}`;
  image.src = URL.createObjectURL(await response.blob());
  image.onload = () => URL.revokeObjectURL(image.src);
  viewer.replaceChildren(image);
  job.textContent = `Foreground mask ${name}`;
}
