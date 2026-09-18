# Duplicate Hash Split Analysis

**Generated:** 2026-09-18T08:13:45.667399

## Summary
- Total duplicate groups: 123
- Total duplicate images: 251
- Groups entirely within one split: 60
- Groups crossing splits: 63

## Action Taken
WARNING: 63 groups cross splits. Manual review needed.

## Note
Patient-level leakage cannot be fully assessed from the available APTOS metadata. The APTOS 2019 dataset does not provide patient identifiers; each image ID is unique per examination.

## Cross-Split Groups (first 10)
- Hash `028d27ff645a...`: ['012a242ac6ff', '1f07dae3cadb'] in ['train', 'val']
- Hash `02e855cb5c1e...`: ['0161338f53cc', 'cac40227d3b2'] in ['train', 'test']
- Hash `c04aefbe905a...`: ['034cb07a550f', 'c8d2d32f7f29'] in ['train', 'test']
- Hash `b5b0a2781910...`: ['04ac765f91a1', '3044022c6969'] in ['val', 'test']
- Hash `cf167b308c15...`: ['05a5183c92d0', '63a03880939c'] in ['train', 'test']
- Hash `89ecd9da08e2...`: ['0ac436400db4', 'fda39982a810'] in ['train', 'val']
- Hash `a6e3a43a938c...`: ['1006345f70b7', '435d900fa7b2'] in ['train', 'test']
- Hash `f5a81fd1e590...`: ['11242a67122d', '65c958379680'] in ['train', 'test']
- Hash `972c61131194...`: ['14515b8f19b6', 'ba2624883599'] in ['val', 'test']
- Hash `197303f49434...`: ['14e3f84445f7', 'f0f89314e860'] in ['train', 'test']