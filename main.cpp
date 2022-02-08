#include "lab-segmentation.h"
#include <iostream>

int main()
{
  try
  {
    runSegmentationLab();
  }
  catch (const std::exception& e)
  {
    std::cerr << "Caught exception:\n" << e.what() << std::endl;
  }
  catch (...)
  {
    std::cerr << "Caught unknown exception" << std::endl;
  }

  return EXIT_SUCCESS;
}
