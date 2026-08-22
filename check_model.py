import joblib

model = joblib.load("hotelguard_final_model.pkl")

print("MODEL TYPE:")
print(type(model))

print("\nPIPELINE STEPS:")
print(model.named_steps)

print("\nPREPROCESSOR:")
print(model.named_steps["preprocessor"])

print("\nMODEL LOADED SUCCESSFULLY!")