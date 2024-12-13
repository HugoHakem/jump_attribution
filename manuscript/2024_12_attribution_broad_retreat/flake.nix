{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    systems.url = "github:nix-systems/default";
    devenv.url = "github:cachix/devenv";
  };

  nixConfig = {
    extra-trusted-public-keys = "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
  };

  outputs = { self, nixpkgs, devenv, systems, ... } @ inputs:
    let
      forEachSystem = nixpkgs.lib.genAttrs (import systems);
    in
    {
      packages = forEachSystem (system: {
        devenv-up = self.devShells.${system}.default.config.procfileScript;
      });

      devShells = forEachSystem
        (system:
          let
            pkgs = import nixpkgs {
              system = system;
              config.allowUnfree = true;
            };

            mpkgs = import inputs.nixpkgs_master {
              system = system;
              config.allowUnfree = true;
            };
          in
          {
            default = devenv.lib.mkShell {
              inherit inputs pkgs;
              modules = [
                {
                  stdenv = pkgs.clangStdenv;
                  packages = with pkgs; [
                    texliveFull
                    # python311Packages.pygments
                    emacsPackages.citeproc
                    emacsPackages.org
                    emacsPackages.org-contrib
                    # texliveBasic.withPackages("minted")
	                  emacs29
                    line-awesome
                    crimson-pro
                    noto-fonts
                    freefont_ttf

                    # Qr codes
                    qrtool
                    imagemagick
                  ];
                  enterShell = ''
                  export FONTCONFIG_FILE=$(nix-build -E 'let pkgs = import <nixpkgs> { }; in pkgs.makeFontsConf { fontDirectories = [ pkgs.freefont_ttf ]; }')
                  '';
                  
                }
              ];
              
            };
          });
    };
}
