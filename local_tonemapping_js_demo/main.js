import * as THREE from 'https://unpkg.com/three@0.138.0/build/three.module.js';

import { EXRLoader } from 'https://unpkg.com/three@0.138.0/examples/jsm/loaders/EXRLoader.js';
import { GUI } from 'https://unpkg.com/three@0.138.0/examples/jsm/libs/lil-gui.module.min.js';
import { ShaderPass } from 'https://unpkg.com/three@0.138.0/examples/jsm/postprocessing/ShaderPass.js';
import { CopyShader } from 'https://unpkg.com/three@0.138.0/examples/jsm/shaders/CopyShader.js';

const passthroughVS = /* glsl */`
varying vec2 vUv;

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`;

const LuminanceShader = {
    uniforms: {
        'tOriginal': { value: null },
        'shadows': { value: 0.5 },
        'highlights': { value: 2.0 },
        'exposure': { value: 1.0 },
    },
    vertexShader: passthroughVS,
    fragmentShader: /* glsl */`
		uniform sampler2D tOriginal;
        uniform float shadows;
        uniform float highlights;
        uniform float exposure;

		varying vec2 vUv;

		void main() {
            // Tonemap three syntetic exposures and produce their luminances.
            vec3 inpColor = texture2D( tOriginal, vUv ).xyz * exposure;
            float highlights = sqrt(dot(clamp(ACESFilmicToneMapping(inpColor * highlights), 0.0, 1.0), vec3(0.1,0.7,0.2)));
            float midtones = sqrt(dot(clamp(ACESFilmicToneMapping(inpColor), 0.0, 1.0), vec3(0.1,0.7,0.2)));
            float shadows = sqrt(dot(clamp(ACESFilmicToneMapping(inpColor * shadows), 0.0, 1.0), vec3(0.1,0.7,0.2)));
			vec3 cols = vec3(highlights, midtones, shadows);
			gl_FragColor = vec4(cols, 1.0);
		}`
};


const ExposureWeightsShader = {
    uniforms: {
        'tDiffuse': { value: null },
        'sigmaSq': { value: 1000.0 },
    },
    vertexShader: passthroughVS,
    fragmentShader: /* glsl */`
		uniform sampler2D tDiffuse;
        uniform float sigmaSq;

		varying vec2 vUv;
		void main() {
            // Compute the synthetic exposure weights.
			vec3 diff = texture2D(tDiffuse, vUv).xyz - vec3(0.5);
            vec3 weights = vec3(exp(-0.5 * diff * diff * sigmaSq));
            weights /= dot(weights, vec3(1.0)) + 0.00001;
			gl_FragColor = vec4(weights, 1.0);
		}`
};

const BlendShader = {
    uniforms: {
        'tExposures': { value: null },
        'tWeights': { value: null },
    },
    vertexShader: passthroughVS,
    fragmentShader: /* glsl */`
		uniform sampler2D tExposures;
        uniform sampler2D tWeights;
		varying vec2 vUv;

		void main() {
            // Blend the exposures based on the blend weights.
			vec3 weights = texture2D(tWeights, vUv).xyz;
            vec3 exposures = texture2D(tExposures, vUv).xyz;
            weights /= dot(weights, vec3(1.0)) + 0.0001;
			gl_FragColor = vec4(vec3(dot(exposures * weights, vec3(1.0))), 1.0);
		}`
};

const BlendLaplacianShader = {
    uniforms: {
        'tExposures': { value: null },
        'tWeights': { value: null },
        'tExposuresCoarser': { value: null },
        'tAccumSoFar': { value: null },
        'boostLocalContrast': { value: 1.0 },
    },
    vertexShader: passthroughVS,
    fragmentShader: /* glsl */`
		uniform sampler2D tExposures;
        uniform sampler2D tWeights;
        uniform sampler2D tExposuresCoarser;
        uniform sampler2D tAccumSoFar;
        uniform float boostLocalContrast;
		varying vec2 vUv;

		void main() {
            // Blend the Laplacians based on exposure weights.
            float accumSoFar = texture2D( tAccumSoFar, vUv ).x;
            vec3 laplacians = texture2D(tExposures, vUv).xyz - texture2D(tExposuresCoarser, vUv).xyz;
            vec3 weights = texture2D(tWeights, vUv).xyz * (boostLocalContrast > 0.0 ? abs(laplacians) + 0.00001 : vec3(1.0));
            weights /= dot(weights, vec3(1.0)) + 0.00001;
            float laplac = dot(laplacians * weights, vec3(1.0));
			gl_FragColor = vec4(vec3(accumSoFar + laplac), 1.0);
		}`
};

const FinalCombinePassShader = {
    uniforms: {
        'tDiffuse': { value: null },
        'tOriginal': { value: null },
        'tOriginalMip': { value: null },
        'mipPixelSize': { value: null },
        'exposure': { value: 1.0 },

    },
    vertexShader: passthroughVS,
    fragmentShader: /* glsl */`
		uniform sampler2D tDiffuse;
        uniform sampler2D tOriginal;
        uniform sampler2D tOriginalMip;
        uniform vec4 mipPixelSize;
		varying vec2 vUv;
        uniform float exposure;

		void main() {
            // Guided upsampling.
            // See https://bartwronski.com/2019/09/22/local-linear-models-guided-filter/
            float momentX = 0.0;
            float momentY = 0.0;
            float momentX2 = 0.0;
            float momentXY = 0.0;
            float ws = 0.0;
            for (int dy = -1; dy <= 1; dy += 1) {
                for (int dx = -1; dx <= 1; dx += 1) {
                    float x = texture2D(tOriginalMip, vUv + vec2(dx, dy) * mipPixelSize.zw).y;
                    float y = texture2D(tDiffuse, vUv + vec2(dx, dy) * mipPixelSize.zw).x;
                    float w = exp(-0.5 * float(dx*dx + dy*dy) / (0.7*0.7));
                    momentX += x * w;
                    momentY += y * w;
                    momentX2 += x * x * w;
                    momentXY += x * y * w;
                    ws += w;
                }
            }
            momentX /= ws;
            momentY /= ws;
            momentX2 /= ws;
            momentXY /= ws;
            float A = (momentXY - momentX * momentY) / (max(momentX2 - momentX * momentX, 0.0) + 0.00001);
            float B = momentY - A * momentX;
            
            // Apply local exposure adjustment as a crude multiplier on all RGB channels.
            // This is... generally pretty wrong, but enough for the demo purpose.
            vec3 texel = texture2D(tDiffuse, vUv).xyz;
            vec3 texelOriginal = sqrt(max(ACESFilmicToneMapping(texture2D(tOriginal, vUv).xyz * exposure), 0.0));
            float luminance = dot(texelOriginal.xyz, vec3(0.1,0.7,0.2)) + 0.00001;
            float finalMultiplier = max(A * luminance + B, 0.0) / luminance;
            // This is a hack to prevent super dark pixels getting boosted by a lot and showing compression artifacts.
            float lerpToUnityThreshold = 0.007;
            finalMultiplier = luminance > lerpToUnityThreshold ? finalMultiplier : 
                mix(1.0, finalMultiplier, (luminance / lerpToUnityThreshold) * (luminance / lerpToUnityThreshold));
            vec3 texelFinal = sqrt(max(ACESFilmicToneMapping(texture2D(tOriginal, vUv).xyz * exposure * finalMultiplier), 0.0));
			gl_FragColor = vec4(texelFinal, 1.0);
		}`
};


const params = {
    enable_ltm: true,
    boost_local_contrast: false,
    exposure: 0.7,
    mip: 6.0,
    shadows: 1.5,
    highlights: 2.0,
    display_mip: 2,
    esposure_preference_sigma: 5.0,
    save: saveAsImage
};

let renderer, effectCopy, effectFinalCombine, effectWeights, effectBlend, effectBlendLaplacian,
    effectLuminance, mips, mipsWeights, mipsAssemble, texture;

init();

function init() {
    renderer = new THREE.WebGLRenderer();
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);
    renderer.toneMapping = THREE.LinearToneMapping;
    renderer.outputEncoding = THREE.LinearEncoding;

    let w, h;
    new EXRLoader()
        .load('veranda_2k.exr', function (tex, textureData) {
            const material = new THREE.MeshBasicMaterial({ map: tex });
            w = textureData.width;
            h = textureData.height;
            texture = tex;

            let sharedProps = { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, format: THREE.RGBAFormat, type: THREE.HalfFloatType };
            let assembleProps = { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, format: THREE.RedFormat, type: THREE.HalfFloatType };

            mips = []
            mipsWeights = []
            mipsAssemble = []
            while (w > 1 && h > 1) {
                mips.push(new THREE.WebGLRenderTarget(w, h, sharedProps));
                mipsWeights.push(new THREE.WebGLRenderTarget(w, h, sharedProps));
                mipsAssemble.push(new THREE.WebGLRenderTarget(w, h, assembleProps));
                w = w / 2;
                h = h / 2;
            }
            render();
        });

    effectCopy = new ShaderPass(CopyShader);
    effectFinalCombine = new ShaderPass(FinalCombinePassShader);
    effectLuminance = new ShaderPass(LuminanceShader, null);
    effectWeights = new ShaderPass(ExposureWeightsShader);
    effectBlend = new ShaderPass(BlendShader, null);
    effectBlendLaplacian = new ShaderPass(BlendLaplacianShader, null);

    const gui = new GUI();

    gui.add(params, 'enable_ltm').onChange(render);
    gui.add(params, 'boost_local_contrast').onChange(render);
    gui.add(params, 'exposure', 0, 4, 0.01).onChange(render);
    gui.add(params, 'shadows', 0, 4, 0.1).onChange(render);
    gui.add(params, 'highlights', 0, 4, 0.1).onChange(render);
    gui.add(params, 'mip', 0, 9, 1.0).onChange(render);
    gui.add(params, 'display_mip', 0, 8, 1.0).onChange(render);
    gui.add(params, 'esposure_preference_sigma', 0, 10, 1.0).onChange(render);
    gui.add(params, 'save');
    gui.open();
}

window.addEventListener('resize', onWindowResize, false);

function onWindowResize() {
    renderer.setSize(window.innerWidth, window.innerHeight);
    render();
}

function saveAsImage() {
    render();
    var imgData;
    var strDownloadMime = "image/octet-stream";
    var strMime = "image/jpeg";
    imgData = renderer.domElement.toDataURL(strMime);
    imgData = imgData.replace(strMime, strDownloadMime);
    var link = document.createElement('a');
    if (typeof link.download === 'string') {
        document.body.appendChild(link);
        link.download = 'snapshot.jpg';
        link.href = imgData;
        link.click();
        document.body.removeChild(link); //remove the link when done
    } else {
        location.replace(uri);
    }
}

function render() {
    // Compute the luminances of synthetic exposures.
    effectLuminance.uniforms['exposure'].value = params.exposure;
    effectLuminance.uniforms['shadows'].value = params.enable_ltm ? Math.pow(2, params.shadows) : 1.0;
    effectLuminance.uniforms['highlights'].value = params.enable_ltm ? Math.pow(2, -params.highlights) : 1.0;
    effectLuminance.uniforms['tOriginal'].value = texture;
    effectLuminance.render(renderer, mips[0]);

    // Compute the local weights of synthetic exposures.
    effectWeights.uniforms['sigmaSq'].value = params.esposure_preference_sigma * params.esposure_preference_sigma;
    effectWeights.render(renderer, mipsWeights[0], mips[0]);
    for (let i = 0; i < mips.length - 1; i++) {
        effectCopy.render(renderer, mips[i + 1], mips[i]);
        effectCopy.render(renderer, mipsWeights[i + 1], mipsWeights[i]);
    }

    // Blend the coarsest level - Gaussian.
    effectBlend.uniforms['tExposures'].value = mips[params.mip].texture;
    effectBlend.uniforms['tWeights'].value = mipsWeights[params.mip].texture;
    effectBlend.render(renderer, mipsAssemble[params.mip]);
    for (let i = params.mip; i > params.display_mip; i--) {
        // Blend the finer levels - Laplacians.
        effectBlendLaplacian.uniforms['tExposures'].value = mips[i - 1].texture;
        effectBlendLaplacian.uniforms['tExposuresCoarser'].value = mips[i].texture;
        effectBlendLaplacian.uniforms['boostLocalContrast'].value = params.boost_local_contrast;
        effectBlendLaplacian.uniforms['tWeights'].value = mipsWeights[i - 1].texture;
        effectBlendLaplacian.uniforms['tAccumSoFar'].value = mipsAssemble[i].texture;

        effectBlendLaplacian.render(renderer, mipsAssemble[i - 1]);
    }

    // Perform guided upsampling and output the final RGB image.
    let display_mip = Math.min(params.display_mip, params.mip);
    effectFinalCombine.renderToScreen = true;
    effectFinalCombine.uniforms['tOriginal'].value = texture;
    effectFinalCombine.uniforms['mipPixelSize'].value = new THREE.Vector4(
        mips[display_mip].width,
        mips[display_mip].height,
        1.0 / mips[display_mip].width,
        1.0 / mips[display_mip].height);
    effectFinalCombine.uniforms['tOriginalMip'].value = mips[display_mip].texture;
    effectFinalCombine.uniforms['exposure'].value = params.exposure;
    effectFinalCombine.render(renderer, null, mipsAssemble[display_mip]);
}
