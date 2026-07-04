# Recording the demo GIF

Open `tmp/iris_rf.html` in your browser and record a ~10 second GIF showing:

1. The growth animation on load (~3s)
2. Hover a tree (tooltip appears)
3. Switch to "🍂 Autumn" season
4. Click "Trace" on a sample (badges appear)
5. Double-click a tree (detail modal opens)

## Tools

- **macOS**: Use [Gifcap](https://gifcap.dev) (browser-based, no install)
- **CLI**: `ffmpeg -i input.mov -vf "fps=12,scale=700:-1" -loop 0 demo.gif`
- **App**: [LICEcap](https://www.cockos.com/licecap/) or [Kap](https://getkap.co)

Save the output as `assets/demo.gif` (keep it under 5MB for GitHub).
