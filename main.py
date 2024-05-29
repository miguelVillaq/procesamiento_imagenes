import segmentacion_functions as sf
import denoising as dn
import intensity_standarisation as st
import border as br
import tkinter as tk
from tkinter import filedialog
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import registration as rt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import SimpleITK as sitk
import laplacian 

class ImageProcessingApp(tk.Tk):
    
    def __init__(self):
        super().__init__()
        self.title("Image Processing App")

        self.image_data = None
        self.original_img = None
        self.segmentation_data = None
        self.current_slice = 0  # Track current slice for line drawing
        self.lines_red = []  # List to store line coordinates
        self.lines_green = []  # List to store line coordinates
        self.image_width = None
        self.image_height = None
        self.ax = None
        self.img_slice = None
        
        # Variables para guardar las coordenadas del dibujo
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None

        self.create_menu()
        self.create_image_display()
        self.create_algorithm_form()
        self.create_navigation_bar()
        
    def create_menu(self):
        menu_bar = tk.Menu(self)

        # Menú Archivo
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Abrir", command=self.open_image)
        file_menu.add_command(label="Guardar", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Reset", command=self.reset_image)
        file_menu.add_command(label="Salir", command=self.quit)
        menu_bar.add_cascade(label="Archivo", menu=file_menu)

        # Menú segmentación.
        algorithm_menu = tk.Menu(menu_bar, tearoff=0)
        algorithm_menu.add_command(label="Umbralización", command=self.show_umbralization_form)
        algorithm_menu.add_command(label="Isodata", command=self.show_isodata_form)
        algorithm_menu.add_command(label="Region growing", command=self.show_rg_form)
        algorithm_menu.add_command(label="K-Means", command=self.show_kmeans_form)
        algorithm_menu.add_command(label="Proyecto", command=self.show_proyecto_form)
        menu_bar.add_cascade(label="Segmentacion", menu=algorithm_menu)
        
        # Menú denoising.
        algorithm_menu = tk.Menu(menu_bar, tearoff=0)
        algorithm_menu.add_command(label="Mean denoising", command=self.show_mean_d_form)
        algorithm_menu.add_command(label="Median denoising", command=self.show_median_d_form)
        menu_bar.add_cascade(label="Denoising", menu=algorithm_menu)
        
        # Menú intensity standarisation.
        algorithm_menu = tk.Menu(menu_bar, tearoff=0)
        algorithm_menu.add_command(label="Intensity rescaling", command=self.show_intensity_form)
        algorithm_menu.add_command(label="Z-score", command=self.show_z_score_form)
        algorithm_menu.add_command(label="Histogram matching", command=self.show_histogram_form)
        algorithm_menu.add_command(label="White strip", command=self.show_white_form)
        menu_bar.add_cascade(label="Intensity standarisation", menu=algorithm_menu)
        
        # Menú border detection.
        algorithm_menu = tk.Menu(menu_bar, tearoff=0)
        algorithm_menu.add_command(label="First derivative", command=self.show_first_derivative)
        algorithm_menu.add_command(label="Second derivative", command=self.show_second_derivative)
        algorithm_menu.add_command(label="Difference filter", command=self.show_difference_filter)
        menu_bar.add_cascade(label="Border detection", menu=algorithm_menu)
        
        algorithm_menu = tk.Menu(menu_bar, tearoff=0)
        algorithm_menu.add_command(label="Registration", command=self.show_registration)
        menu_bar.add_cascade(label="Registration", menu=algorithm_menu)
        
        self.config(menu=menu_bar)
        
    def create_image_display(self):
        self.right_frame = tk.Frame(self)
        self.right_frame.pack()

        self.figure = plt.Figure(figsize=(4,4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.right_frame)
        self.canvas.draw
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # Bind mouse events for line drawing
        self.canvas.get_tk_widget().bind("<Button-1>", self.start_drawing)
        self.canvas.get_tk_widget().bind("<Button-3>", self.start_drawing)
        self.canvas.get_tk_widget().bind("<B1-Motion>", self.draw_red)
        self.canvas.get_tk_widget().bind("<B3-Motion>", self.draw_green)
        self.canvas.get_tk_widget().bind("<ButtonRelease-1>", self.stop_drawing)
        self.canvas.get_tk_widget().bind("<ButtonRelease-3>", self.stop_drawing) 
        
    def save_image(self):
        # Ask user for the desired filename
        filename = tk.filedialog.asksaveasfilename(
            filetypes=[("NIfTI files", "*.nii"), ("All files", "*.*")],
            defaultextension=".nii",
        )
        # Check if filename is valid
        if filename:
            # Save the image using the provided filename
            img = self.image_data.astype(np.float32)
            segmented_image = nib.Nifti1Image(img, affine=self.original_img.affine)
            nib.save(segmented_image, filename)
            print(f"Image saved successfully to {filename}")
        else:
            print("No filename provided. Image not saved.")
    
    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii"), ("All files", "*.*")])
        if file_path:
            self.original_img = nib.load(file_path)
            self.image_data = self.original_img.get_fdata()
            
            self.navigation_scale.config(to=self.image_data.shape[2]-1)
            
            self.update_image_display()
            
    def update_image_display(self, event=None):
        if self.image_data is not None:
            current_slice = int(self.navigation_scale.get())
            axis = self.navigation_var.get()
            if(len(self.image_data.shape) > 2):
                
                if axis == "X":
                    image_slice = self.image_data[current_slice, :, :]
                elif axis == "Y":
                    image_slice = self.image_data[:, current_slice, :]
                else:  # axis == "Z"
                    image_slice = self.image_data[:, :, current_slice]
            else:
                image_slice = self.image_data

            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            self.img_slice = image_slice
            self.ax.imshow(image_slice)
            
            #self.solapar_coordenadas()
            
            self.canvas.draw()

    def solapar_coordenadas(self):
            fig = self.figure
            ax = self.ax

            # Obtener la posición del subplot
            position = ax.get_position()
            fig_size = fig.get_size_inches()

            # Obtener la posición relativa del subplot (entre 0 y 1)
            x0 = position.x0
            y0 = position.y0

            # Obtener el ancho y alto de la figura en pulgadas
            fig_width = fig_size[0]
            fig_height = fig_size[1]

            # Obtener la densidad de puntos por pulgada (dpi) de la pantalla
            dpi = fig.get_dpi()

            # Calcular el ancho y alto del subplot en píxeles
            width_pixels = fig_width * dpi * (position.x1 - position.x0)
            height_pixels = fig_height * dpi * (position.y1 - position.y0)
            
            # Alto y ancho de la figura en píxeles.
            width_pixels_f = fig_width * dpi 
            height_pixels_f = fig_height * dpi 
            
            dif_x = round(width_pixels_f - width_pixels)
            dif_y = round(height_pixels_f - height_pixels)
            x0_img = int(dif_x/2) + 5
            x1_img = int(width_pixels + x0_img)
            y0_img = int(dif_y/2) + 2
            y1_img = int(height_pixels + y0_img)
            
            factor_conversion = self.img_slice.shape[0] / height_pixels
            
            def aux(lista):
                lista[0] = int((lista[0] - x0_img) * factor_conversion)
                lista[1] = int((lista[1] - y0_img) * factor_conversion)
                return lista
            
            self.lines_red = np.array(list(map(aux, self.lines_red)))
            self.lines_green = np.array(list(map(aux, self.lines_green)))

            #print(factor_conversion)
            # print(f"Dimensiones del subplot en píxeles:")
            # print(f"X0 {x0_img} , Y0 {y0_img}, X1 {x1_img} , y1 {y1_img}")
            # print(f"Ancho: {width_pixels:.2f} píxeles")
            # print(f"Alto: {height_pixels:.2f} píxeles")
            # print(f"AnchoF: {width_pixels_f:.2f} píxeles")
            # print(f"AltoF: {height_pixels_f:.2f} píxeles")
            

    def start_drawing(self, event):
        self.start_x = event.x
        self.start_y = event.y
        
    def stop_drawing(self, event):
        self.end_x = event.x
        self.end_y = event.y
        #print(f"ANTES Red: {self.lines_red} \n Green: {self.lines_green}")
        #self.solapar_coordenadas()
        #print(f"DESPUÉS Red: {self.lines_red} \n Green: {self.lines_green}")
             
    def draw_red(self, event):
        if self.start_x and self.start_y:
            self.end_x = event.x
            self.end_y = event.y
            self.lines_red.append([self.end_x, self.end_y])
            self.canvas.get_tk_widget().create_line(self.start_x, self.start_y, self.end_x, self.end_y, width=5, fill="red", tags="red")      

    def draw_green(self, event):
        if self.start_x and self.start_y:
            self.end_x = event.x
            self.end_y = event.y
            self.lines_green.append([self.end_x, self.end_y])
            self.canvas.get_tk_widget().create_line(self.start_x, self.start_y, self.end_x, self.end_y, width=5, fill="green",tags="green")
        
    def create_algorithm_form(self):
        self.algorithm_frame = tk.Frame(self)
        self.algorithm_frame.pack(pady=12)

        self.algorithm_label = tk.Label(self.algorithm_frame, text="Parámetros de Algoritmo:")
        self.algorithm_label.grid(row=0, column=0, columnspan=2, pady=5)

        self.algorithm_entry_label_1 = tk.Label(self.algorithm_frame, text="Parámetro:")
        self.algorithm_entry_label_1.grid(row=1, column=0, padx=5)

        self.algorithm_entry_1 = tk.Entry(self.algorithm_frame)
        self.algorithm_entry_1.grid(row=1, column=1, padx=5)
        
        self.algorithm_entry_label_2 = tk.Label(self.algorithm_frame, text="Parámetro:")
        self.algorithm_entry_label_2.grid(row=2, column=0, padx=5)

        self.algorithm_entry_2 = tk.Entry(self.algorithm_frame)
        self.algorithm_entry_2.grid(row=2, column=1, padx=5)
    
        self.algorithm_entry_label_3 = tk.Label(self.algorithm_frame, text="Parámetro:")
        self.algorithm_entry_label_3.grid(row=3, column=0, padx=5)

        self.algorithm_entry_3 = tk.Entry(self.algorithm_frame)
        self.algorithm_entry_3.grid(row=3, column=1, padx=5)
        
        self.algorithm_entry_label_4 = tk.Label(self.algorithm_frame, text="Parámetro:")
        self.algorithm_entry_label_4.grid(row=4, column=0, padx=5)

        self.algorithm_entry_4 = tk.Entry(self.algorithm_frame)
        self.algorithm_entry_4.grid(row=4, column=1, padx=5)
        
        self.algorithm_entry_label_5 = tk.Label(self.algorithm_frame, text="Parámetro:")
        self.algorithm_entry_label_5.grid(row=5, column=0, padx=5)

        self.algorithm_entry_5 = tk.Entry(self.algorithm_frame)
        self.algorithm_entry_5.grid(row=5, column=1, padx=5)
        
        self.run_algorithm_button = tk.Button(self.algorithm_frame, text="Ejecutar", command=self.run_algorithm)
        self.run_algorithm_button.grid(row=6, column=0, columnspan=2, pady=5)
        
    def create_navigation_bar(self):
        self.navigation_frame = tk.Frame(self)
        self.navigation_frame.pack(pady=10)

        self.navigation_label = tk.Label(self.navigation_frame, text="Seleccionar eje:")
        self.navigation_label.grid(row=0, column=0, padx=5)

        self.navigation_var = tk.StringVar(self)
        self.navigation_var.set("Z")
        self.navigation_menu = tk.OptionMenu(self.navigation_frame, self.navigation_var, "X", "Y", "Z")
        self.navigation_menu.grid(row=0, column=1, padx=5)

        
        self.navigation_scale = tk.Scale(self.navigation_frame, from_=0, to=100, orient="horizontal", command=self.update_image_display)
        self.navigation_scale.grid(row=0, column=2, padx=5)
  
    def show_umbralization_form(self):
        self.algorithm_label.config(text="Parámetros de Umbralización:")
        self.algorithm_entry_label_1.config(text="Umbral:")
        self.algorithm_entry_1.config(state='normal')
        self.algorithm_entry_2.config(state='disabled')
        self.algorithm_entry_3.config(state='disabled')
        self.algorithm_entry_4.config(state='disabled')
        self.algorithm_entry_5.config(state='disabled')

    def show_isodata_form(self):
        self.algorithm_label.config(text="Parámetros de Isodata:")
        self.algorithm_entry_label_1.config(text="Umbral:")
        self.algorithm_entry_label_2.config(text="Tolerancia:")
        self.algorithm_entry_1.config(state='normal')
        self.algorithm_entry_2.config(state='normal')
        self.algorithm_entry_3.config(state='disabled')
        self.algorithm_entry_4.config(state='disabled')
        self.algorithm_entry_5.config(state='disabled')

    def show_kmeans_form(self):
        self.algorithm_label.config(text="Parámetros de K-Means:")
        self.algorithm_entry_label_1.config(text="K:")
        self.algorithm_entry_label_2.config(text="Númnero iteraciones:")
        self.algorithm_entry_1.config(state='normal')
        self.algorithm_entry_2.config(state='normal')
        self.algorithm_entry_3.config(state='disabled')
        self.algorithm_entry_4.config(state='disabled')
        self.algorithm_entry_5.config(state='disabled')

    def show_rg_form(self):
        self.algorithm_label.config(text="Parámetros de Region growing:")
        self.algorithm_entry_label_1.config(text="Tolerancia:")                   
        self.algorithm_entry_label_2.config(text="X:") 
        self.algorithm_entry_label_3.config(text="Y:")   
        self.algorithm_entry_label_4.config(text="Z:")
        self.algorithm_entry_label_5.config(text="Iteraciones profundidad:")
        self.algorithm_entry_1.config(state='normal')
        self.algorithm_entry_2.config(state='normal')
        self.algorithm_entry_3.config(state='normal')
        self.algorithm_entry_4.config(state='normal')
        self.algorithm_entry_5.config(state='normal')

    def show_mean_d_form(self):
        self.algorithm_label.config(text="Parámetros de denoising mean:")
        self.algorithm_entry_label_1.config(text="Z:")
        self.algorithm_entry_1.config(state='normal')
        self.algorithm_entry_label_2.config(text="Tolerancia:")
        self.algorithm_entry_2.config(state='normal')
        self.algorithm_entry_label_3.config(text="Profundidad:")
        self.algorithm_entry_3.config(state='normal')
        self.algorithm_entry_4.config(state='disabled')
        self.algorithm_entry_5.config(state='disabled')
    
    def show_median_d_form(self):
        self.algorithm_label.config(text="Parámetros de denoising median:")
        self.algorithm_entry_label_1.config(text="Z:")
        self.algorithm_entry_1.config(state='normal')
        self.algorithm_entry_label_2.config(text="Tolerancia:")
        self.algorithm_entry_2.config(state='normal')
        self.algorithm_entry_label_3.config(text="Profundidad:")
        self.algorithm_entry_3.config(state='normal')
        self.algorithm_entry_4.config(state='disabled')
        self.algorithm_entry_5.config(state='disabled')
          
    def show_intensity_form(self):
        self.algorithm_label.config(text="Parámetros de Intensity rescaling:")
        self.algorithm_entry_1.config(state='disabled')
        self.algorithm_entry_2.config(state='disabled')
        self.algorithm_entry_3.config(state='disabled')
        self.algorithm_entry_4.config(state='disabled')
        self.algorithm_entry_5.config(state='disabled')

    def show_z_score_form(self):
        self.algorithm_label.config(text="Parámetros de z-score:")
        self.algorithm_entry_1.config(state='disabled')
        self.algorithm_entry_2.config(state='disabled')
        self.algorithm_entry_3.config(state='disabled')
        self.algorithm_entry_4.config(state='disabled')
        self.algorithm_entry_5.config(state='disabled')

    def show_histogram_form(self):
        self.algorithm_label.config(text="Parámetros de histogram matching:")
        self.algorithm_entry_label_1.config(text="K percentiles:")
        self.algorithm_entry_1.config(state='normal')
        self.algorithm_entry_2.config(state='disabled')
        self.algorithm_entry_3.config(state='disabled')
        self.algorithm_entry_4.config(state='disabled')
        self.algorithm_entry_5.config(state='disabled')
        
    def show_white_form(self):
        self.algorithm_label.config(text="Parámetros de white strip:")
        self.algorithm_entry_1.config(state='disabled')
        self.algorithm_entry_2.config(state='disabled')
        self.algorithm_entry_3.config(state='disabled')
        self.algorithm_entry_4.config(state='disabled')
        self.algorithm_entry_5.config(state='disabled')
    
    def show_second_derivative(self):
        self.algorithm_label.config(text="Parámetros de second derivative:")
        self.algorithm_entry_label_1.config(text="Slide:")
        self.algorithm_entry_1.config(state='normal')
        self.algorithm_entry_label_2.config(text="Eje:")
        self.algorithm_entry_2.config(state='normal')
        self.algorithm_entry_label_3.config(text="Umbral:")
        self.algorithm_entry_3.config(state='normal')
        self.algorithm_entry_4.config(state='disabled')
        self.algorithm_entry_5.config(state='disabled')
        
    def show_difference_filter(self):
        self.algorithm_label.config(text="Parámetros de difference filter:")
        self.algorithm_entry_label_1.config(text="Slide:")
        self.algorithm_entry_1.config(state='normal')
        self.algorithm_entry_label_2.config(text="Eje:")
        self.algorithm_entry_2.config(state='normal')
        self.algorithm_entry_label_3.config(text="Umbral:")
        self.algorithm_entry_3.config(state='normal')
        self.algorithm_entry_4.config(state='disabled')
        self.algorithm_entry_5.config(state='disabled')
        
    def show_first_derivative(self):
        self.algorithm_label.config(text="Parámetros de first derivative:")
        self.algorithm_entry_label_1.config(text="Slide:")
        self.algorithm_entry_1.config(state='normal')
        self.algorithm_entry_label_2.config(text="Eje:")
        self.algorithm_entry_2.config(state='normal')
        self.algorithm_entry_label_3.config(text="Umbral:")
        self.algorithm_entry_3.config(state='normal')
        self.algorithm_entry_4.config(state='disabled')
        self.algorithm_entry_5.config(state='disabled')
        
    def show_registration(self):
        self.algorithm_label.config(text="Parámetros de registration:")
        self.algorithm_entry_label_1.config(text="Steps_opt:")
        self.algorithm_entry_1.config(state='normal')
        self.algorithm_entry_label_2.config(text="Tolerance_opt:")
        self.algorithm_entry_2.config(state='normal')
        self.algorithm_entry_label_3.config(text="Repetitions_opt:")
        self.algorithm_entry_3.config(state='normal')
        self.algorithm_entry_label_4.config(text="Name_fixed_img:")
        self.algorithm_entry_4.config(state='normal')
        self.algorithm_entry_label_5.config(text="Name_mov_img:")
        self.algorithm_entry_5.config(state='normal')
        
    def show_proyecto_form(self):
        self.algorithm_label.config(text="Parámetros de proyecto:")
        self.algorithm_entry_label_1.config(text="Beta:")
        self.algorithm_entry_1.config(state='normal')
        self.algorithm_entry_2.config(state='disabled')
        self.algorithm_entry_3.config(state='disabled')
        self.algorithm_entry_4.config(state='disabled')
        self.algorithm_entry_5.config(state='disabled')
              
    def run_algorithm(self):
        if self.algorithm_label.cget("text") == "Parámetros de Umbralización:":
            umbral = float(self.algorithm_entry_1.get())
            if self.image_data is not None:
                self.image_data = sf.threshold(self.image_data, umbral)
        elif self.algorithm_label.cget("text") == "Parámetros de Isodata:":
            umbral = float(self.algorithm_entry_1.get())
            tolerancia = float(self.algorithm_entry_2.get())
            if self.image_data is not None:
                self.image_data = sf.isodata(self.image_data, umbral, tolerancia)
        elif self.algorithm_label.cget("text") == "Parámetros de Region growing:":
            tolerancia = int(self.algorithm_entry_1.get())
            x = int(self.algorithm_entry_2.get())
            y = int(self.algorithm_entry_3.get())
            z = int(self.algorithm_entry_4.get())
            iter = int(self.algorithm_entry_5.get())
            if self.image_data is not None:
                self.image_data = sf.region_growing3D(self.image_data,tolerancia,x,y,z, iter)
        elif self.algorithm_label.cget("text") == "Parámetros de K-Means:":
            k = int(self.algorithm_entry_1.get())
            num_iter = int(self.algorithm_entry_2.get())
            if self.image_data is not None:
                self.image_data = sf.kmeans(self.image_data, k, num_iter)
        elif self.algorithm_label.cget("text") == "Parámetros de denoising mean:":
            z = int(self.algorithm_entry_1.get())
            tol = int(self.algorithm_entry_2.get())
            dep = int(self.algorithm_entry_3.get())
            func = dn.mean
            if self.image_data is not None:
                self.image_data = dn.denoising_img(self.image_data, z, tol, dep, func)
        elif self.algorithm_label.cget("text") == "Parámetros de denoising median:":
            z = int(self.algorithm_entry_1.get())
            tol = int(self.algorithm_entry_2.get())
            dep = int(self.algorithm_entry_3.get())
            func = dn.median
            if self.image_data is not None:
                self.image_data = dn.denoising_img(self.image_data, z, tol, dep, func)
        elif self.algorithm_label.cget("text") == "Parámetros de Intensity rescaling:":
            if self.image_data is not None:
                self.image_data = st.intensity_rescaling(self.image_data)
        elif self.algorithm_label.cget("text") == "Parámetros de z-score:":
            if self.image_data is not None:
                self.image_data = st.z_score(self.image_data)
        elif self.algorithm_label.cget("text") == "Parámetros de histogram matching:":
            k = int(self.algorithm_entry_1.get())
            trainData = nib.load("img\mni152.nii")
            trainData = trainData.get_fdata()
            if self.image_data is not None:
                self.image_data = st.n_matching(self.image_data,trainData, k)
        elif self.algorithm_label.cget("text") == "Parámetros de white strip:":
            if self.image_data is not None:
                self.image_data = st.white_stripe(self.image_data)
        elif self.algorithm_label.cget("text") == "Parámetros de second derivative:":
            slide = int(self.algorithm_entry_1.get())
            eje = str(self.algorithm_entry_2.get())
            umbral = int(self.algorithm_entry_3.get())
            if self.image_data is not None:
                self.image_data = br.derivada_segundo_orden(self.image_data, slide,eje,umbral)
        elif self.algorithm_label.cget("text") == "Parámetros de difference filter:":
            slide = int(self.algorithm_entry_1.get())
            eje = str(self.algorithm_entry_2.get())
            umbral = int(self.algorithm_entry_3.get())
            if self.image_data is not None:
                self.image_data = br.dif_filtro(self.image_data, slide,eje,umbral)
        elif self.algorithm_label.cget("text") == "Parámetros de first derivative:":
            slide = int(self.algorithm_entry_1.get())
            eje = str(self.algorithm_entry_2.get())
            umbral = int(self.algorithm_entry_3.get())
            if self.image_data is not None:
                self.image_data = br.derivada_primer_orden(self.image_data, slide,eje,umbral)
        elif self.algorithm_label.cget("text") == "Parámetros de registration:":
            steps = float(self.algorithm_entry_1.get())
            tol = float(self.algorithm_entry_2.get())
            reps = int(self.algorithm_entry_3.get())
            fixed = f"img\{str(self.algorithm_entry_4.get())}.nii"
            mov = f"img\{str(self.algorithm_entry_5.get())}.nii"
            if self.image_data is not None:
                self.image_data, registration_img = rt.registration(fixed, mov, steps, tol, reps)
                sitk.WriteImage(registration_img, "img\imagen_registrada.nii")
        elif self.algorithm_label.cget("text") == "Parámetros de proyecto:":
            if self.image_data is not None:
                #print(f"ANTES Red: {self.lines_red} \n Green: {self.lines_green}")
                self.solapar_coordenadas()
                #print(f"DESPUÉS Red: {self.lines_red} \n Green: {self.lines_green}")
                beta = float(self.algorithm_entry_1.get())
                self.image_data = laplacian.ejecutar(self.img_slice, beta, self.lines_red, self.lines_green)
                self.canvas.get_tk_widget().delete("red")
                self.canvas.get_tk_widget().delete("green")

        self.update_image_display()
    
    def reset_values(self):
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.lines_red = []  
        self.lines_green = []  
    
    def reset_image(self):
            self.image_data = None
            self.figure.clear()
            self.canvas.get_tk_widget().delete("red")
            self.canvas.get_tk_widget().delete("green")
            self.reset_values()
            self.canvas.draw()
            self.open_image()
            
app = ImageProcessingApp()
app.mainloop()