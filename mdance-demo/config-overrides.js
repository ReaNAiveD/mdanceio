module.exports = function override(config, env) {
    const wasmExtensionRegExp = /\.wasm$/;
    config.resolve.extensions.push(".wasm");
    config.experiments = {
        asyncWebAssembly: true,
    };
    config.module.rules.forEach((rule) => {
        (rule.oneOf || []).forEach((oneOf) => {
            if (oneOf.type === "asset/resource") {
                oneOf.exclude.push(wasmExtensionRegExp);
            }
        });
    });

    return config;
}
