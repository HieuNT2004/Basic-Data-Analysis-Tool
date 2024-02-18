'''
Scripted by Hieu
Chú thích:
Định lượng: thường là kiểu số
Định tính: thường là kiểu chữ
std: độ lệch chuẩn
CV = độ lệch chuẩn / trung bình
'''
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

def browse_file():
    filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if filename:
        entry_path.delete(0, tk.END)
        entry_path.insert(0, filename)
        display_csv_info(filename)

def display_csv_info(filename):
    try:
        df = pd.read_csv(filename)
        df.dropna(how='all', inplace=True)
        df.drop_duplicates(inplace=True)
        csv_info_text.delete(1.0, tk.END)
        csv_info_text.insert(tk.END, f"CSV File Information:\n\n")
        csv_info_text.insert(tk.END, f"{df.to_string(index=False)}\n")
    except Exception as e:
        tk.messagebox.showerror("Error", f"Error loading CSV file: {e}")

def analyze_plot():
    file_path = entry_path.get()
    if not file_path:
        tk.messagebox.showerror("Error", "Please select a CSV file.")
        return

    try:
        df = pd.read_csv(file_path)
        df.dropna(how='all', inplace=True)
        df.drop_duplicates(inplace=True)
    except Exception as e:
        tk.messagebox.showerror("Error", f"Error loading CSV file: {e}")
        return

    analysis_type = var_plot_type.get()

    x_var = entry_x_type.get()
    y_var = entry_y_type.get()
    hue_var = entry_hue_type.get()
    agg_var = entry_agg_type.get()

    if analysis_type == "Boxplot":
        if not x_var and not y_var and not hue_var:
            sns.boxplot(data=df)
        elif x_var and not y_var and not hue_var:
            sns.boxplot(x=x_var, data=df)
        elif x_var and y_var and not hue_var:
            sns.boxplot(x=x_var, y=y_var, data=df)
        elif x_var and y_var and hue_var:
            sns.boxplot(x=x_var, y=y_var, hue=hue_var, data=df)
        plt.title("Boxplot")
        plt.show()
    elif analysis_type == "Pie Chart":
        if x_var and y_var and agg_var:
            gb = df.groupby([x_var])[y_var].agg([agg_var])
            labels = gb.index
            data = list(gb[agg_var])
            colors = sns.color_palette('pastel')
            plt.pie(data, labels=labels, colors=colors,
                    autopct='%1.1f%%', shadow=True)
            plt.show()
    elif analysis_type == "Line Chart":
        if x_var and y_var and hue_var:
            sns.lineplot(x=x_var, y=y_var, hue=hue_var, data=df)
            plt.show()
        elif x_var and y_var and hue_var and agg_var:
            sns.lineplot(x=x_var, y=y_var, data=df, hue=hue_var, estimator=agg_var)
            plt.show()
    elif analysis_type == "Bar":
        if x_var and y_var and hue_var:
            sns.barplot(x=x_var, y=y_var, hue=hue_var, data=df)
            plt.show()
        elif x_var and y_var:
            sns.barplot(x=x_var, y=y_var, data=df)
            plt.show()
    elif analysis_type == "Histogram":
        if x_var:
            plt.hist(df[x_var].dropna(), bins='auto', color='blue', alpha=0.7)
            plt.xlabel(x_var)
            plt.ylabel('Frequency')
            plt.title('Histogram')
            plt.show()
    elif analysis_type == "Scatter":
        if x_var and y_var and hue_var:
            sns.scatterplot(x=x_var, y=y_var, hue=hue_var, data=df)
            plt.show()
        elif x_var and y_var:
            plt.scatter(df[x_var], df[y_var])
            plt.xlabel(x_var)
            plt.ylabel(y_var)
            plt.title('Scatter Plot')
            plt.show()
    elif analysis_type == "Count Plot":
        if x_var and hue_var and agg_var:
            gb = df.groupby([x_var, hue_var])[agg_var].agg("count").reset_index()
            sns.barplot(x=x_var, y=agg_var, hue=hue_var, data=gb)
            plt.show()
        elif x_var and agg_var:
            sns.barplot(x=x_var, y=agg_var, data=df, estimator=np.count_nonzero)
            plt.show()
    elif analysis_type == "Pair Plot":
        if hue_var:
            sns.pairplot(df, hue=hue_var)
            plt.show()
        else:
            sns.pairplot(df)
            plt.show()
    elif analysis_type == "Heatmap to visualize missing data":
        sns.heatmap(df.isna(), cmap='YlGnBu', cbar_kws={'label': 'Missing Data'})
        plt.title("Heatmap to visualize missing data")
        plt.show()
def analyze_inference():
    file_path = entry_path.get()
    if not file_path:
        tk.messagebox.showerror("Error", "Please select a CSV file.")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        tk.messagebox.showerror("Error", f"Error loading CSV file: {e}")
        return

    analysis_type = var_inference_type.get()

    if analysis_type == "Linear Regression":
        linear_regression(df, entry_x_type.get(), entry_y_type.get())
    elif analysis_type == "One Sample T-Test":
        x_var = entry_x_type.get()
        alpha = 0.05
        if not x_var:
            tk.messagebox.showerror("Error", "Please specify the X variable.")
            return

        t_test(df[x_var])
    elif analysis_type == "Two Sample T-Test":
        x_var = entry_x_type.get()
        y_var = entry_y_type.get()
        alpha = 0.05
        if not x_var or not y_var:
            tk.messagebox.showerror("Error", "Please specify both X and Y variables.")
            return
        t_test(df[x_var], df[y_var])
    elif analysis_type == "One Sample Z-Test":
        x_var = entry_x_type.get()
        alpha = 0.05
        if not x_var:
            tk.messagebox.showerror("Error", "Please specify the X variable.")
            return

        z_test(df[x_var])
    elif analysis_type == "Pearson's Correlation Test":
        x_var = entry_x_type.get()
        y_var = entry_y_type.get()
        alpha = 0.05
        if not x_var or not y_var:
            tk.messagebox.showerror("Error", "Please specify both X and Y variables.")
            return
        correlation_test(df[x_var], df[y_var])
    elif analysis_type == "FISHER-TEST(F-TEST)":
        fisher_test(df, entry_x_type.get(), entry_y_type.get())
    elif analysis_type == "Chi-square Test":
        chi_square_test(df, entry_x_type.get(), entry_y_type.get())
    elif analysis_type == "One way Anova":
        one_way_anova(df, entry_x_type.get(), entry_y_type.get())

def t_test(x_data, y_data=None):
    if y_data is None:
        mean_value = np.mean(x_data)
        t_statistic, p_value = stats.ttest_1samp(x_data, popmean=mean_value)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"One Sample T-Test Result:\n\n")
        result_text.insert(tk.END, f"Mean Value: {mean_value}\n")
        result_text.insert(tk.END, f"T-Statistic: {t_statistic}\n")
        result_text.insert(tk.END, f"P-Value: {p_value}\n")
        result_text.insert(tk.END, f"Alpha: 0.05\n")
        if p_value < 0.05:
            result_text.insert(tk.END, "Conclusion: Reject null hypothesis (H0)\n")
        else:
            result_text.insert(tk.END, "Conclusion: Fail to reject null hypothesis (H0)\n")
    else:
        t_statistic, p_value = stats.ttest_ind(x_data, y_data)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Two Sample T-Test Result:\n\n")
        result_text.insert(tk.END, f"T-Statistic: {t_statistic}\n")
        result_text.insert(tk.END, f"P-Value: {p_value}\n")
        result_text.insert(tk.END, f"Alpha: 0.05\n")
        if p_value < 0.05:
            result_text.insert(tk.END, "Conclusion: Reject null hypothesis (H0)\n")
        else:
            result_text.insert(tk.END, "Conclusion: Fail to reject null hypothesis (H0)\n")

def z_test(x_data):
    mean_value = np.mean(x_data)
    z_statistic, p_value = stats.ztest(x_data, value=mean_value)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"One Sample Z-Test Result:\n\n")
    result_text.insert(tk.END, f"Mean Value: {mean_value}\n")
    result_text.insert(tk.END, f"Z-Statistic: {z_statistic}\n")
    result_text.insert(tk.END, f"P-Value: {p_value}\n")
    result_text.insert(tk.END, f"Alpha: 0.05\n")
    if p_value < 0.05:
        result_text.insert(tk.END, "Conclusion: Reject null hypothesis (H0)\n")
    else:
        result_text.insert(tk.END, "Conclusion: Fail to reject null hypothesis (H0)\n")

def correlation_test(x_data, y_data):
    correlation_coefficient, p_value = stats.pearsonr(x_data, y_data)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"Pearson's Correlation Test Result:\n\n")
    result_text.insert(tk.END, f"Correlation Coefficient: {correlation_coefficient}\n")
    result_text.insert(tk.END, f"P-Value: {p_value}\n")
    result_text.insert(tk.END, f"Alpha: 0.05\n")
    if p_value < 0.05:
        result_text.insert(tk.END, "Conclusion: Reject null hypothesis (H0)\n")
    else:
        result_text.insert(tk.END, "Conclusion: Fail to reject null hypothesis (H0)\n")

def fisher_test(df, x_var, y_var):
    crossdata = pd.crosstab(df[x_var], [df[y_var]])
    odd_ratio, p_value = stats.fisher_exact(crossdata)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"Fisher Test Results:\n\n")
    result_text.insert(tk.END, f"Odd ratio: {odd_ratio}\n")
    result_text.insert(tk.END, f"P-value: {p_value}\n")


def chi_square_test(df, x_var, y_var):
    crossdata = pd.crosstab(df[x_var], [df[y_var]])
    stats, p, dof, expected = chi2_contingency(crossdata)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"Chi-square Test Results:\n\n")
    result_text.insert(tk.END, f"P-value: {p}\n")
    if p <= 0.05:
        result_text.insert(tk.END, f"Dependent (reject H0)\n")
    else:
        result_text.insert(tk.END, f"Independent (H0 holds true)\n")

def one_way_anova(df, x_var, y_var):
    model = ols(f'{y_var} ~ C({x_var})', data=df).fit()
    aov_table = sm.stats.anova_lm(model, typ=1)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"One way ANOVA Results:\n\n")
    result_text.insert(tk.END, f"{aov_table}\n")


def linear_regression(df, x_var, y_var):
    x_with_constant = sm.add_constant(df[[x_var]].values)
    y = df[[y_var]].values

    result = sm.OLS(y, x_with_constant).fit()

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"Linear Regression Results for {y_var} ~ {x_var}:\n\n")
    result_text.insert(tk.END, result.summary())


root = tk.Tk()
root.title("Data Analysis Tool -- make by HieuNguyen")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = 1280
window_height = 720
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2

root.minsize(width=1280, height=720)
root.geometry(f"{window_width}x{window_height}+{x}+{y}")


label_path = tk.Label(root, text="CSV File:")
entry_path = tk.Entry(root, width=50)
button_browse = tk.Button(root, text="Browse", command=browse_file)

label_plot_type = tk.Label(root, text="Plot Type:")
var_plot_type = tk.StringVar(root)
var_plot_type.set("Boxplot")
option_menu_plot_type = tk.OptionMenu(root, var_plot_type, "Boxplot", "Pie Chart", "Line Chart", "Heatmap to visualize missing data", "Bar", "Histogram", "Scatter", "Count Plot", "Pair Plot")

button_analyze_plot = tk.Button(root, text="Analyze Plot", command=analyze_plot)


label_inference_type = tk.Label(root, text="Inference Type:")
var_inference_type = tk.StringVar(root)
var_inference_type.set("Linear Regression")
option_menu_inference_type = tk.OptionMenu(root, var_inference_type, "Linear Regression", "One Sample T-Test",
                                           "Two Sample T-Test",
                                           "One Sample Z-Test",
                                           "Pearson's Correlation Test",
                                           "FISHER-TEST(F-TEST)",
                                           "Chi-square Test",
                                           "One way Anova")

button_analyze_inference = tk.Button(root, text="Analyze Inference", command=analyze_inference)


result_frame = tk.Frame(root, bd=2, relief="sunken", width=300, height=300)
result_label = tk.Label(result_frame, text="Inference Results:")
result_text = tk.Text(result_frame, wrap="word", height=10, width=50)

csv_info_frame = tk.Frame(root, bd=2, relief="sunken", width=300, height=300)
csv_info_label = tk.Label(csv_info_frame, text="CSV Information:")
csv_info_text = tk.Text(csv_info_frame, wrap="word", height=10, width=50)

###########

variable_frame = tk.Frame(root)
variable_frame.grid(row=4, column=2, columnspan=2, padx=10, pady=10, sticky="se")


label_x_type = tk.Label(variable_frame, text="X Variable:")
entry_x_type = tk.Entry(variable_frame, width=20)


label_y_type = tk.Label(variable_frame, text="Y Variable:")
entry_y_type = tk.Entry(variable_frame, width=20)


label_hue_type = tk.Label(variable_frame, text="Hue Variable:")
entry_hue_type = tk.Entry(variable_frame, width=20)

label_agg_type = tk.Label(variable_frame, text="Agg Variable:")
entry_agg_type = tk.Entry(variable_frame, width=20)

label_x_type.grid(row=0, column=0, padx=5, pady=5)
entry_x_type.grid(row=0, column=1, padx=5, pady=5)
label_y_type.grid(row=1, column=0, padx=5, pady=5)
entry_y_type.grid(row=1, column=1, padx=5, pady=5)
label_hue_type.grid(row=2, column=0, padx=5, pady=5)
entry_hue_type.grid(row=2, column=1, padx=5, pady=5)
label_agg_type.grid(row=3, column=0, padx=5, pady=5)
entry_agg_type.grid(row=3, column=1, padx=5, pady=5)


result_frame.config(height=150)


label_path.grid(row=0, column=0, padx=5, pady=5)
entry_path.grid(row=0, column=1, padx=5, pady=5)
button_browse.grid(row=0, column=2, padx=0, pady=5)


label_plot_type.grid(row=1, column=0, padx=5, pady=5)
option_menu_plot_type.grid(row=1, column=1, padx=5, pady=5)
button_analyze_plot.grid(row=1, column=2, padx=5, pady=5)

label_inference_type.grid(row=2, column=0, padx=5, pady=5)
option_menu_inference_type.grid(row=2, column=1, padx=5, pady=5)
button_analyze_inference.grid(row=2, column=2, padx=5, pady=5)


csv_info_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")
csv_info_label.pack(side="top", fill="x")
csv_info_text.pack(side="top", fill="both", expand=True)

result_frame.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")
result_label.pack(side="top", fill="x")
result_text.pack(side="top", fill="both", expand=True)

for i in range(3):
    root.grid_columnconfigure(i, weight=1)
for i in range(5):
    root.grid_rowconfigure(i, weight=1)

root.mainloop()
