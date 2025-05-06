from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  
import matplotlib.pyplot as plt  
import seaborn as sns  
import joblib  

# Ensure the model is available
try:  
    final_model = optimized_models["Optimized Random Forest"]  # Or replace with your optimized model
except KeyError:  
    print("❌ Optimized model not found. Verify the model name in `optimized_models`.")  
    exit()  

# Predictions on the Test set  
y_pred_test = final_model.predict(X_test)  

# Compute Evaluation Metrics  
mse = mean_squared_error(y_test, y_pred_test)  
rmse = mean_squared_error(y_test, y_pred_test, squared=False)  
mae = mean_absolute_error(y_test, y_pred_test)  
r2 = r2_score(y_test, y_pred_test)  

# Print results with a clean format  
print("\n📊 Final Model Performance (Test Set):")  
print(f"  - MSE: ${mse:,.2f}")  
print(f"  - RMSE: ${rmse:,.2f}")  
print(f"  - MAE: ${mae:,.2f}")  
print(f"  - R²: {r2:.3f}")  

# Visualize Actual vs Predicted
plt.figure(figsize=(8, 4))  
sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.6)  
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--', linewidth=1)  # Diagonal line for perfect prediction
plt.title("Actual vs. Predicted Charges (Test Set)")  
plt.xlabel("Actual Charges (USD)")  
plt.ylabel("Predicted Charges (USD)")  
plt.show()

# Save the final model for deployment
joblib.dump(final_model, "optimized_random_forest_model.pkl")
