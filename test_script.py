from data_loader import load_data, train_model, FEATURES
print('Features:', FEATURES)
df = load_data()
print('Shape:', df.shape)
missing = [f for f in FEATURES if f not in df.columns]
print('Missing features:', missing)
bundle = train_model(df)
print('R2:', bundle['r2'])
print('All checks passed')
