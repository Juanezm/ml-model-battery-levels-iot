from src.utils.cleaning import df_to_image
from PIL import Image as PILImage

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Spacer, Image


def add_text_to_story(text, style, story, space_after=12):
    """
    Add a styled Paragraph to a story.

    Args:
    text (str): The text to add.
    style: The style to apply to the text.
    story (list): The story to append to.
    space_after (int, optional): The height of the Spacer to append after the Paragraph. Defaults to 12.
    """
    text = text.replace('\n', '<br/>')
    p = Paragraph(text, style)
    story.append(p)
    story.append(Spacer(1, space_after))

def add_section_title_to_story(title, story, style=None, space_after=12):
    """
    Add a styled section title to a story.

    This function creates a Paragraph with the given title and style, and appends it to the story.
    It also appends a Spacer with the given height after the Paragraph. If no style is provided,
    it uses the 'Heading2' style from the sample style sheet.

    Args:
    title (str): The title to add.
    style: The style to apply to the title. If None, use 'Heading2' from the sample style sheet.
    story (list): The story to append to.
    space_after (int, optional): The height of the Spacer to append after the Paragraph. Defaults to 12.
    """
    if style is None:
        style = getSampleStyleSheet()['Heading2']

    p = Paragraph(title, style)
    story.append(p)
    story.append(Spacer(1, space_after))

def add_image_to_story(img_filename, story, space_after=12, max_img_width=400, max_img_height=700):
    """
    Add a section with a image to a story.

    Args:
    img_filename (str): The base name of the image file. The image will be named '{img_filename}.png'.
    story (list): The story to append to.
    space_after (int, optional): The height of the Spacer to append after the image. Defaults to 12.
    max_img_width (int, optional): The maximum width of the image. The image is resized to fit within this width
        while maintaining its aspect ratio. Defaults to 400.
    max_img_height (int, optional): The maximum height of the image. The image is resized if it exceeds this height.
        Defaults to 700.
    """
    # Open the image file with PIL to get its size
    with PILImage.open(f'{img_filename}.png') as img:
        w, h = img.size

    # Calculate the new height to maintain aspect ratio
    new_height = max_img_width * h / w

    # If the new height exceeds the maximum height, recalculate both width and height
    if new_height > max_img_height:
        aspect_ratio = w / h
        new_height = max_img_height
        max_img_width = new_height * aspect_ratio

    # Create a ReportLab Image object and add it to the story
    img = Image(f'{img_filename}.png', width=max_img_width, height=new_height)
    img.hAlign = 'CENTER'
    story.append(img)

    # Add a spacer after the image
    story.append(Spacer(1, space_after))


def add_table_to_story(df, img_filename, story):
    """
    Add a section with a table (in form of an image) to a story.

    Args:
      df (pandas.DataFrame): The DataFrame to convert to an image.
      img_filename (str): The base name of the image file to create. The image will be named '{img_filename}.png'.
      story (list): The story to append to.
    """
    # Convert the DataFrame to an image
    df_to_image(df, img_filename)

    # Add the converted image to the story
    add_image_to_story(img_filename, story)
