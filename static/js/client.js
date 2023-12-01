// Establish connection with the server
const socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port);

let scene, camera, renderer;
let neuronMeshes = {};
let synapseLines = [];

const initVisualization = () => {
    // Scene setup
    scene = new THREE.Scene();

    // Camera setup
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 5;

    // Renderer setup
    renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040); // Soft white light
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    scene.add(directionalLight);

    // Start the animation loop
    animate();
};

const animate = () => {
    requestAnimationFrame(animate);
    // Any animation-related code goes here
    renderer.render(scene, camera);
};

// Function to create a neuron mesh
const createNeuronMesh = (neuron) => {
    const geometry = new THREE.SphereGeometry(0.05, 32, 32);
    const material = new THREE.MeshPhongMaterial({
        color: new THREE.Color(neuron.activity, neuron.activity, 1 - neuron.activity),
        emissive: new THREE.Color(neuron.activity, neuron.activity, 1 - neuron.activity)
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(neuron.position.x, neuron.position.y, neuron.position.z);
    mesh.neuronId = neuron.id; // Custom property to identify neurons
    return mesh;
};

// Function to create a synapse line
const createSynapseLine = (source, target, weight) => {
    const material = new THREE.LineBasicMaterial({
        color: new THREE.Color(weight, weight, 1 - weight),
        opacity: weight,
        transparent: true
    });
    const points = [source.position, target.position];
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    return new THREE.Line(geometry, material);
};

// Data update handler
socket.on('update_lsm', data => {
    // Clear out old synapses
    synapseLines.forEach(line => scene.remove(line));
    synapseLines = [];

    // Update or create new neuron meshes
    data.neurons.forEach(neuronData => {
        let neuron = neuronMeshes[neuronData.id];
        if (!neuron) {
            neuron = createNeuronMesh(neuronData);
            neuronMeshes[neuronData.id] = neuron;
            scene.add(neuron);
        } else {
            // Update neuron properties if needed
            neuron.position.set(neuronData.position.x, neuronData.position.y, neuronData.position.z);
            neuron.material.color.set(new THREE.Color(neuronData.activity, neuronData.activity, 1 - neuronData.activity));
            neuron.material.emissive.set(new THREE.Color(neuronData.activity, neuronData.activity, 1 - neuronData.activity));
        }
    });

    // Create new synapse lines
    data.synapses.forEach(synapseData => {
        const sourceNeuron = neuronMeshes[synapseData.source];
        const targetNeuron = neuronMeshes[synapseData.target];
        if (sourceNeuron && targetNeuron) {
            const synapse = createSynapseLine(sourceNeuron, targetNeuron, synapseData.weight);
            scene.add(synapse);
            synapseLines.push(synapse);
        }
    });
});

// Call the initialization function
initVisualization();
