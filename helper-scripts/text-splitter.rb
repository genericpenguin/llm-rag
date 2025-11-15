require 'fileutils'

def main(input_file = "pages.txt", output_dir = "pages")
  # Create the output directory if it doesn't exist
  FileUtils.mkdir_p(output_dir)

  # Read and split the text
  text = File.read(input_file)
  pages = text.split(/--- [0-9]+ ---/)

  # Write each page to a separate file
  pages.each_with_index { |p, i| File.write("#{output_dir}/#{i}.txt", p) }

  puts "Split #{input_file} into #{pages.length} pages in #{output_dir}/"
end

# Run the main function if this script is executed directly
if __FILE__ == $0
  input_file = ARGV[0] || "pages.txt"
  output_dir = ARGV[1] || "pages"
  main(input_file, output_dir)
end