const path = require('path');

module.exports = function override(config, env) {
    console.log(env)
    config.resolve.extensions.push('.wasm');
    config.module.rules.forEach(rule => {
        (rule.oneOf || []).forEach(oneOf => {
            if (oneOf.loader && oneOf.loader.indexOf('file-loader') >= 0) {
                oneOf.exclude.push(/\.wasm$/);
            }
        });
    });
    config.module.rules.push({
        test: /\.wasm$/, // only load WASM files (ending in .wasm)
        // only files in our src/ folder
        include: path.resolve(__dirname, "src"),
        use: [{
            // load and use the wasm-loader dictionary
            loader: require.resolve("wasm-loader"),
            options: {}
        }],
    })

    return config;
}
