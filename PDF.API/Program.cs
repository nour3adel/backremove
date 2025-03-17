using Microsoft.AspNetCore.Diagnostics;
using PDF.API.Services;
using PDF.API.settings;

var builder = WebApplication.CreateBuilder(args);


#region  Configure CORS
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAll", 
        builder => builder
            .AllowAnyOrigin()
            .AllowAnyMethod()
            .AllowAnyHeader());
});
#endregion

builder.Services.AddControllers();
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.Configure<BackgroundRemovalSettings>(
    builder.Configuration.GetSection("BackgroundRemovalSettings"));


builder.Services.AddScoped<IBackgroundRemovalService,BackgroundRemovalService>();

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c => {
    c.SwaggerDoc("v1", new Microsoft.OpenApi.Models.OpenApiInfo
    {
        Title = "Background Removal API",
        Version = "v1",
        Description = "API for removing backgrounds from images"
    });
});

// Configure file size limit for image uploads (default is 28.6MB)
builder.Services.Configure<Microsoft.AspNetCore.Http.Features.FormOptions>(options =>
{
    options.MultipartBodyLengthLimit = 10 * 1024 * 1024; // 10MB limit
});

// Ensure TempFiles directory exists
// Ensure Models directory exists
var modelsDir = Path.Combine(Directory.GetCurrentDirectory(), "Saved_Models", "u2net");
if (!Directory.Exists(modelsDir))
{
    Directory.CreateDirectory(modelsDir);
}

// Ensure TempFiles directory exists
var tempFilesDir = Path.Combine(Directory.GetCurrentDirectory(), "TempFiles");
if (!Directory.Exists(tempFilesDir))
{
    Directory.CreateDirectory(tempFilesDir);
}

// Configure IIS Server options for large file uploads
builder.Services.Configure<IISServerOptions>(options =>
{
    options.MaxRequestBodySize = 20 * 1024 * 1024; // 20MB limit
});

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.UseAuthorization();

// Use CORS with the "AllowAll" policy
app.UseCors("AllowAll");

// Add exception handling middleware
app.UseExceptionHandler(exceptionHandlerApp =>
{
    exceptionHandlerApp.Run(async context =>
    {
        context.Response.StatusCode = StatusCodes.Status500InternalServerError;
        context.Response.ContentType = "application/json";

        var exceptionHandlerPathFeature = context.Features.Get<IExceptionHandlerPathFeature>();
        var exception = exceptionHandlerPathFeature?.Error;

        await context.Response.WriteAsJsonAsync(new
        {
            error = "An error occurred while processing your request.",
            detail = exception?.Message
        });
    });
});

app.MapControllers();

app.Run();
